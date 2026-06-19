"""
pipeline.py — Backend job orchestration
========================================
* All pipeline jobs are serialised through a single FIFO queue.
  Only ONE pipeline runs at a time, preventing GPU OOM and resource
  contention when multiple jobs are submitted concurrently.
* After every successful pipeline run an auto-save step copies:
    - original frames  → <classes_dir>/<class_name>/images/
    - cleaned labels   → <classes_dir>/<class_name>/labels/
    - annotated images → <classes_dir>/<class_name>/annotated/
  creating class folders on-the-fly if they don't exist yet.
"""

import itertools
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional

# Ensure backend/ is on sys.path so 'import db' works regardless of CWD
sys.path.insert(0, str(Path(__file__).parent))

import db

ROOT      = Path(__file__).resolve().parent.parent
PIPELINE  = ROOT / "scripts" / "auto_pipeline.py"
VISUALIZE = ROOT / "scripts" / "visualize_labels.py"
PYTHON    = ROOT / "backend" / "venv" / "Scripts" / "python.exe"
if not PYTHON.exists():
    PYTHON = Path(sys.executable)


# ── Global FIFO job queue ─────────────────────────────────────────────────────
# Each item is a callable (no-arg) that executes the full pipeline logic.
_job_queue: queue.Queue = queue.Queue()


def _queue_worker():
    """Single background thread that drains the job queue sequentially."""
    while True:
        task = _job_queue.get()
        try:
            task()
        except Exception as exc:
            # Errors inside the task should already be handled, but guard here
            print(f"[pipeline worker] Unhandled exception: {exc}", flush=True)
        finally:
            _job_queue.task_done()


# Start the single worker daemon thread on module load
_worker_thread = threading.Thread(target=_queue_worker, daemon=True)
_worker_thread.start()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _stream(proc, job_id: str):
    """Stream subprocess stdout line-by-line into the job log."""
    for line in proc.stdout:
        db.append_log(job_id, [line.rstrip()])
    proc.wait()


def _load_class_names(out_dir: Path) -> list:
    """Read class names from 4_dataset/data.yaml."""
    yaml_path = out_dir / "4_dataset" / "data.yaml"
    if not yaml_path.exists():
        return []
    for line in yaml_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip().startswith("names:"):
            raw = line.split("names:", 1)[1].strip()
            return [c.strip().strip("'\"") for c in raw.strip("[]").split(",") if c.strip()]
    return []


def _auto_save_annotated_frames(job_id: str, output_dir: Path):
    """
    After a successful pipeline run, automatically copy every annotated
    frame into the appropriate class directory under annotatted_classes/.

    Layout written per class:
        <classes_dir>/<class_name>/images/    ← original filtered frame
        <classes_dir>/<class_name>/labels/    ← 5-column YOLO .txt
        <classes_dir>/<class_name>/annotated/ ← bounding-box overlay image
    """
    settings    = db.get_settings()
    classes_dir = Path(settings.get("classes_dir", str(ROOT / "annotatted_classes")))
    class_names = _load_class_names(output_dir)

    if not class_names:
        db.append_log(job_id, ["⚠ Auto-save: no class names found in data.yaml — skipping auto-save"])
        return

    labels_dir    = output_dir / "3_labels"
    frames_dir    = output_dir / "2_filtered_frames"
    annotated_dir = output_dir / "5_annotated"

    if not labels_dir.exists():
        db.append_log(job_id, ["⚠ Auto-save: 3_labels/ not found — skipping auto-save"])
        return

    saved_count  = 0
    skipped_empty = 0

    label_files = sorted(labels_dir.glob("*.txt"))
    db.append_log(job_id, [f"📂 Auto-save: scanning {len(label_files)} label file(s) across {len(class_names)} class(es)…"])

    for lbl_path in label_files:
        raw_lines = lbl_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
        box_lines = [l for l in raw_lines if l.strip()]

        if not box_lines:
            skipped_empty += 1
            continue

        # Determine which classes appear in this label file
        seen_classes: set[str] = set()
        clean_lines: list[str] = []
        for line in box_lines:
            parts = line.split()
            if not parts:
                continue
            try:
                cls_id = int(float(parts[0]))
            except ValueError:
                continue
            # 5-column only (strip optional 6th confidence column)
            clean_lines.append(" ".join(parts[:5]))
            if cls_id < len(class_names):
                seen_classes.add(class_names[cls_id])

        stem = lbl_path.stem

        # Find the source frame (any common image extension)
        src_img: Optional[Path] = None
        for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG", ".bmp"):
            candidate = frames_dir / (stem + ext)
            if candidate.exists():
                src_img = candidate
                break

        # Find the annotated overlay image
        ann_img: Optional[Path] = None
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = annotated_dir / (stem + ext)
            if candidate.exists():
                ann_img = candidate
                break

        for class_name in seen_classes:
            dst_images    = classes_dir / class_name / "images"
            dst_labels    = classes_dir / class_name / "labels"
            dst_annotated = classes_dir / class_name / "annotated"
            for d in (dst_images, dst_labels, dst_annotated):
                d.mkdir(parents=True, exist_ok=True)

            # Copy original frame
            if src_img:
                shutil.copy2(src_img, dst_images / src_img.name)

            # Write clean label
            (dst_labels / lbl_path.name).write_text(
                "\n".join(clean_lines), encoding="utf-8"
            )

            # Copy annotated overlay
            if ann_img:
                shutil.copy2(ann_img, dst_annotated / ann_img.name)

            saved_count += 1

        # Mark as saved in the DB so the review UI reflects it
        if ann_img:
            db.mark_saved(job_id, ann_img.name)

    db.append_log(job_id, [
        f"✅ Auto-save complete: {saved_count} frame-class pair(s) saved, "
        f"{skipped_empty} empty label(s) skipped."
    ])


# ── Public API ────────────────────────────────────────────────────────────────

def run_pipeline(
    job_id: str,
    input_dir: Path,
    output_dir: Path,
    classes: str,
    skip_training: bool,
    conf: str = "0.35",
    fps: str = "1.0",
    labeler: str = "yolo",
    device: str = "0",
    yolo_model: str = "",
    input_type: str = "videos",
    roboflow_model_id: str = "",
    skip_filter: bool = False,
    filter_preset: str = "none",
):
    """Enqueue a pipeline job. Returns immediately; execution is serialised."""

    def _execute():
        db.set_status(job_id, "running")
        db.append_log(job_id, ["▶ Job started (dequeued from pipeline queue)"])
        try:
            # ── Roboflow cloud inference path ──────────────────────────────────
            if labeler == "roboflow":
                _run_roboflow_pipeline(
                    job_id=job_id,
                    input_dir=input_dir,
                    output_dir=output_dir,
                    conf=float(conf),
                    fps=fps,
                    roboflow_model_id=roboflow_model_id,
                    input_type=input_type,
                )
                # Auto-save after Roboflow run
                _auto_save_annotated_frames(job_id, output_dir)
                return

            # ── Standard local pipeline (YOLO / YOLO-World / Grounding DINO) ──
            cmd = [
                str(PYTHON), str(PIPELINE),
                "--output",  str(output_dir),
                "--classes", classes,
                "--conf",    conf,
                "--fps",     fps,
                "--labeler", labeler,
                "--device",  device,
            ]
            if input_type == "images":
                cmd += ["--input-images", str(input_dir)]
                cmd.append("--skip-filter")   # images are pre-curated — never filter
            else:
                cmd += ["--videos", str(input_dir)]
                if skip_filter:
                    cmd.append("--skip-filter")
            if skip_training:
                cmd.append("--skip-training")
            if labeler in ("yolo", "yolo_world") and yolo_model:
                cmd += ["--yolo-model", yolo_model]
            if filter_preset != "none":
                cmd += ["--filter-preset", filter_preset]

            db.append_log(job_id, ["▶ " + " ".join(cmd)])
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                cwd=str(ROOT),
            )
            _stream(proc, job_id)

            if proc.returncode != 0:
                db.set_status(job_id, "error")
                db.append_log(job_id, [f"❌ Pipeline exited with code {proc.returncode}"])
                return

            db.append_log(job_id, ["▶ Running visualize_labels..."])
            proc2 = subprocess.Popen(
                [str(PYTHON), str(VISUALIZE), "--pipeline-output", str(output_dir)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
                cwd=str(ROOT),
            )
            _stream(proc2, job_id)

            if proc2.returncode != 0:
                db.set_status(job_id, "error")
                db.append_log(job_id, [f"❌ visualize_labels exited with code {proc2.returncode}"])
                return

            # ── Auto-save annotated frames to class folders ────────────────────
            _auto_save_annotated_frames(job_id, output_dir)

            db.set_status(job_id, "ready")
            db.append_log(job_id, ["✓ Ready for review!"])

        except Exception as exc:
            db.set_status(job_id, "error")
            db.append_log(job_id, [f"Exception: {exc}"])

    # Mark as queued immediately so the UI shows the right state
    db.set_status(job_id, "queued")
    queue_pos = _job_queue.qsize() + 1
    db.append_log(job_id, [f"🕐 Job queued (position ~{queue_pos} in queue)"])
    _job_queue.put(_execute)


def _run_roboflow_pipeline(
    job_id: str,
    input_dir: Path,
    output_dir: Path,
    conf: float,
    fps: str,
    roboflow_model_id: str,
    input_type: str,
):
    """
    Roboflow-based pipeline:
      1. Extract frames from video (if needed) → 2_filtered_frames/
      2. Call Roboflow hosted inference on each frame → 3_labels/
      3. Run visualize_labels.py to draw bounding boxes → 5_annotated/
    """
    import roboflow_utils as rf

    try:
        db.append_log(job_id, [f"🌐 Roboflow Cloud Inference: model={roboflow_model_id}"])

        settings = db.get_settings()
        api_key  = settings.get("roboflow_api_key", "").strip()
        if not api_key:
            db.set_status(job_id, "error")
            db.append_log(job_id, ["❌ No Roboflow API key found. Set it in Settings first."])
            return
        if not roboflow_model_id:
            db.set_status(job_id, "error")
            db.append_log(job_id, ["❌ No Roboflow model ID specified (e.g. my-project/3)"])
            return

        # ── Step 1: Extract frames if input is a video ───────────────────────
        frames_dir = output_dir / "2_filtered_frames"
        if not frames_dir.exists() or not any(frames_dir.iterdir()):
            frames_dir.mkdir(parents=True, exist_ok=True)
            if input_type == "images":
                img_exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
                all_imgs = sorted(itertools.chain.from_iterable(
                    input_dir.glob(ext) for ext in img_exts
                ))
                for img in all_imgs:
                    dest = frames_dir / img.name
                    if not dest.exists():
                        shutil.copy2(img, dest)
                copied = len(list(frames_dir.iterdir()))
                db.append_log(job_id, [f"  Copied {copied} images"])
            else:
                db.append_log(job_id, [f"  Extracting frames at {fps} FPS..."])
                video_files = (
                    list(input_dir.glob("*.mp4")) + list(input_dir.glob("*.avi"))
                    + list(input_dir.glob("*.mov")) + list(input_dir.glob("*.mkv"))
                )
                if not video_files:
                    db.set_status(job_id, "error")
                    db.append_log(job_id, ["❌ No video files found in input directory"])
                    return
                for video_file in video_files:
                    db.append_log(job_id, [f"  Extracting from {video_file.name}..."])
                    pattern = str(frames_dir / f"{video_file.stem}_%06d.jpg")
                    proc = subprocess.Popen(
                        ["ffmpeg", "-i", str(video_file), "-vf", f"fps={fps}", "-q:v", "2", pattern, "-y"],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, encoding="utf-8", errors="replace",
                    )
                    for _ in proc.stdout:
                        pass  # suppress ffmpeg verbose output
                    proc.wait()
                total_frames = len(list(frames_dir.glob("*.jpg")))
                db.append_log(job_id, [f"  Extracted {total_frames} frames"])

        frame_count = len(list(itertools.chain.from_iterable(
            frames_dir.glob(ext)
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
        )))
        if frame_count == 0:
            db.set_status(job_id, "error")
            db.append_log(job_id, ["❌ No frames to process"])
            return

        # ── Step 2: Roboflow inference → YOLO labels ─────────────────────────
        labels_dir = output_dir / "3_labels"
        labels_dir.mkdir(parents=True, exist_ok=True)
        db.append_log(job_id, [f"  Sending {frame_count} frames to Roboflow API..."])

        stats, class_names = rf.autolabel_roboflow(
            image_dir=str(frames_dir),
            label_dir=str(labels_dir),
            model_id=roboflow_model_id,
            api_key=api_key,
            conf_threshold=conf,
        )

        db.append_log(job_id, [
            f"  ✓ Labelled: {stats['labelled']}  Empty: {stats['empty']}  Errors: {stats['error']}",
            f"  Classes: {', '.join(class_names) if class_names else '(none detected)'}",
        ])

        if stats["labelled"] == 0 and stats["error"] > 0:
            db.set_status(job_id, "error")
            db.append_log(job_id, ["❌ All frames failed — check API key and model ID"])
            return

        # ── Step 2b: Write data.yaml so visualize_labels knows class names ────
        dataset_dir = output_dir / "4_dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        yaml_path = dataset_dir / "data.yaml"
        names_str = ", ".join(f"'{c}'" for c in class_names) if class_names else "'object'"
        yaml_path.write_text(
            f"# Roboflow auto-labelled dataset\n"
            f"path: {dataset_dir}\n"
            f"train: images/train\nval: images/val\n\n"
            f"nc: {len(class_names) if class_names else 1}\n"
            f"names: [{names_str}]\n",
            encoding="utf-8"
        )
        db.append_log(job_id, [f"  Dataset YAML written with classes: {', '.join(class_names) if class_names else 'object'}"])

        # ── Step 3: Visualize labels → 5_annotated/ ──────────────────────────
        db.append_log(job_id, ["▶ Visualizing labels..."])
        proc3 = subprocess.Popen(
            [str(PYTHON), str(VISUALIZE), "--pipeline-output", str(output_dir)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            cwd=str(ROOT),
        )
        for line in proc3.stdout:
            db.append_log(job_id, [line.rstrip()])
        proc3.wait()

        db.set_status(job_id, "ready")
        db.append_log(job_id, ["✓ Roboflow labeling complete — Ready for review!"])

    except Exception as exc:
        db.set_status(job_id, "error")
        db.append_log(job_id, [f"❌ Roboflow pipeline error: {exc}"])


def run_visualize_only(job_id: str, output_dir: Path):
    """Enqueue a visualize-only job (for loading an existing pipeline output)."""

    def _execute():
        db.set_status(job_id, "running")
        try:
            db.append_log(job_id, [f"▶ Running visualize_labels on {output_dir}..."])
            proc = subprocess.Popen(
                [str(PYTHON), str(VISUALIZE), "--pipeline-output", str(output_dir)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
                cwd=str(ROOT),
            )
            _stream(proc, job_id)
            db.set_status(job_id, "ready")
            db.append_log(job_id, ["✓ Ready for review!"])
        except Exception as exc:
            db.set_status(job_id, "error")
            db.append_log(job_id, [f"Exception: {exc}"])

    db.set_status(job_id, "queued")
    queue_pos = _job_queue.qsize() + 1
    db.append_log(job_id, [f"🕐 Job queued (position ~{queue_pos} in queue)"])
    _job_queue.put(_execute)
