import os
import shutil
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

# Ensure backend/ is on sys.path so local imports work regardless of CWD
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

import db
import pipeline as pl

ROOT       = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

app = FastAPI(title="Annotation Tool")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    db.init_db()
    settings = db.get_settings()
    if not settings.get("default_yolo_model"):
        for candidate in [ROOT / "models" / "best_03062026.pt", ROOT / "yolov8m-world.pt"]:
            if candidate.exists():
                db.update_settings({"default_yolo_model": str(candidate)})
                break


# ── Jobs ──────────────────────────────────────────────────────────────────────

@app.get("/api/jobs")
def list_jobs():
    return db.list_jobs()


@app.post("/api/jobs")
async def create_job(
    video:               Optional[UploadFile] = File(default=None),
    image:               Optional[UploadFile] = File(default=None),
    video_path:          str = Form(default=""),
    images_path:         str = Form(default=""),
    existing_output:     str = Form(default=""),
    classes:             str = Form(default=""),
    skip_training:       str = Form(default="true"),
    conf:                str = Form(default="0.35"),
    fps:                 str = Form(default="1.0"),
    labeler:             str = Form(default="yolo"),
    device:              str = Form(default="0"),
    yolo_model:          str = Form(default=""),
    roboflow_model_id:   str = Form(default=""),
    skip_filter:         str = Form(default="false"),
    output_dir_override: str = Form(default=""),
    filter_preset:       str = Form(default="none"),
):
    settings = db.get_settings()
    job_id   = uuid.uuid4().hex[:8]

    # ── Mode: load existing pipeline output ──────────────────────────────────
    if existing_output.strip():
        out_dir = Path(existing_output.strip())
        if not (out_dir / "2_filtered_frames").exists():
            raise HTTPException(400, f"No 2_filtered_frames under: {out_dir}")
        db.create_job(job_id, out_dir.name, str(out_dir), {}, datetime.utcnow().isoformat())
        pl.run_visualize_only(job_id, out_dir)
        return {"job_id": job_id}

    # ── Determine output/input directories ───────────────────────────────────
    base    = Path(output_dir_override.strip() or settings["output_base_dir"])
    out_dir = base / job_id / "output"
    inp_dir = base / job_id / "input"
    inp_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_type = "videos"

    if video and video.filename:
        dest = inp_dir / video.filename
        dest.write_bytes(await video.read())
        job_name = video.filename
    elif video_path.strip():
        src = Path(video_path.strip())
        if not src.exists():
            raise HTTPException(400, f"File not found: {video_path}")
        dest = inp_dir / src.name
        try:
            os.link(src, dest)
        except OSError:
            shutil.copy2(src, dest)
        job_name = src.name
    elif image and image.filename:
        dest = inp_dir / image.filename
        dest.write_bytes(await image.read())
        input_type = "images"
        job_name   = image.filename
    elif images_path.strip():
        src = Path(images_path.strip())
        if not src.is_dir():
            raise HTTPException(400, f"Directory not found: {images_path}")
        inp_dir    = src
        input_type = "images"
        job_name   = src.name
    else:
        raise HTTPException(400, "Provide video_path, images_path, or upload a file")

    effective_model = yolo_model.strip() or settings.get("default_yolo_model", "")
    config = {
        "labeler":           labeler,
        "conf":              conf,
        "fps":               fps,
        "yolo_model":        effective_model,
        "device":            device,
        "input_type":        input_type,
        "roboflow_model_id": roboflow_model_id.strip(),
        "skip_filter":       skip_filter == "true",
        "filter_preset":     filter_preset,
    }
    db.create_job(job_id, job_name, str(out_dir), config, datetime.utcnow().isoformat())
    pl.run_pipeline(
        job_id=job_id,
        input_dir=inp_dir,
        output_dir=out_dir,
        classes=classes or "placeholder",
        skip_training=(skip_training == "true"),
        conf=conf,
        fps=fps,
        labeler=labeler,
        device=device,
        yolo_model=effective_model,
        input_type=input_type,
        roboflow_model_id=roboflow_model_id.strip(),
        skip_filter=(skip_filter == "true"),
        filter_preset=filter_preset,
    )
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str, log_since: int = 0):
    j = db.get_job(job_id)
    if not j:
        raise HTTPException(404)
    ad    = Path(j["output_dir"]) / "5_annotated"
    total = (
        sum(1 for f in ad.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if ad.exists() else 0
    )
    return {
        "status":    j["status"],
        "name":      j["name"],
        "logs":      j["logs"][log_since:],
        "log_count": len(j["logs"]),
        "saved":     len(j["saved_frames"]),
        "skipped":   len(j["skipped_frames"]),
        "total":     total,
        "config":    j["config"],
    }


@app.delete("/api/jobs/{job_id}", status_code=204)
def delete_job(job_id: str):
    if not db.get_job(job_id):
        raise HTTPException(404)
    db.delete_job(job_id)


# ── Review ────────────────────────────────────────────────────────────────────

def _load_class_names(out_dir: Path) -> list[str]:
    """Parse the names list from 4_dataset/data.yaml (same source visualize_labels uses)."""
    yaml_path = out_dir / "4_dataset" / "data.yaml"
    if not yaml_path.exists():
        return []
    for line in yaml_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip().startswith("names:"):
            raw = line.split("names:", 1)[1].strip()
            return [c.strip().strip("'\"") for c in raw.strip("[]").split(",") if c.strip()]
    return []


def _count_classes(label_path: Path, class_names: list[str]) -> dict[str, int]:
    """Count detections per class from a YOLO label file (one box per line)."""
    counts: dict[str, int] = {}
    if not label_path.exists():
        return counts
    for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            cls_id = int(float(parts[0]))
        except ValueError:
            continue
        name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
        counts[name] = counts.get(name, 0) + 1
    return counts


@app.get("/api/jobs/{job_id}/images")
def get_images(job_id: str):
    j = db.get_job(job_id)
    if not j:
        raise HTTPException(404)
    out_dir = Path(j["output_dir"])
    ad      = out_dir / "5_annotated"
    if not ad.exists():
        return {"images": []}
    class_names = _load_class_names(out_dir)
    labels_dir  = out_dir / "3_labels"
    saved   = set(j["saved_frames"])
    skipped = set(j["skipped_frames"])
    images  = []
    for f in sorted(ad.iterdir()):
        if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
            st = "saved" if f.name in saved else "skipped" if f.name in skipped else "pending"
            counts = _count_classes(labels_dir / f"{f.stem}.txt", class_names)
            images.append({
                "name":   f.name,
                "status": st,
                "counts": counts,
                "total":  sum(counts.values()),
            })
    return {"images": images}


@app.get("/api/jobs/{job_id}/image/{filename}")
def serve_image(job_id: str, filename: str):
    j = db.get_job(job_id)
    if not j:
        raise HTTPException(404)
    path = Path(j["output_dir"]) / "5_annotated" / filename
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="image/jpeg")


# Box-drawing palette — matches visualize_labels.COLORS (BGR for OpenCV)
_BOX_COLORS = [
    (0, 255, 0), (0, 165, 255), (255, 0, 0), (0, 255, 255),
    (255, 0, 255), (255, 128, 0), (128, 0, 255),
]


@app.get("/api/jobs/{job_id}/boxes/{filename}")
def serve_boxes_only(job_id: str, filename: str):
    """Render the original frame with bounding boxes only (no class-name text)."""
    import cv2
    j = db.get_job(job_id)
    if not j:
        raise HTTPException(404)
    out  = Path(j["output_dir"])
    stem = Path(filename).stem

    src_img = None
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG", ".bmp"):
        candidate = out / "2_filtered_frames" / (stem + ext)
        if candidate.exists():
            src_img = candidate
            break
    if not src_img:
        raise HTTPException(404, "Source frame not found")

    img = cv2.imread(str(src_img))
    if img is None:
        raise HTTPException(404, "Could not read source frame")
    H, W = img.shape[:2]

    label_path = out / "3_labels" / f"{stem}.txt"
    if label_path.exists():
        for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            try:
                cls_id = int(float(parts[0]))
                cx, cy, bw, bh = (float(p) for p in parts[1:])
            except ValueError:
                continue
            x1, y1 = int((cx - bw / 2) * W), int((cy - bh / 2) * H)
            x2, y2 = int((cx + bw / 2) * W), int((cy + bh / 2) * H)
            color = _BOX_COLORS[cls_id % len(_BOX_COLORS)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        raise HTTPException(500, "Encode failed")
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@app.post("/api/save")
def save_frame(
    job_id:      str = Form(...),
    filename:    str = Form(...),
    class_name:  str = Form(...),
    classes_dir: str = Form(default=""),
):
    j = db.get_job(job_id)
    if not j:
        raise HTTPException(404)
    settings      = db.get_settings()
    c_dir_str     = classes_dir.strip() if classes_dir.strip() else settings["classes_dir"]
    classes_dir   = Path(c_dir_str)
    out           = Path(j["output_dir"])
    stem          = Path(filename).stem
    images_dir    = classes_dir / class_name / "images"
    labels_dir    = classes_dir / class_name / "labels"
    annotated_dir = classes_dir / class_name / "annotated"
    for d in (images_dir, labels_dir, annotated_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Try to find the original source image (any extension)
    # The annotated filename is always .jpg but source may be .jpeg/.png/.JPG etc.
    src_img = None
    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG", ".bmp"]:
        candidate = out / "2_filtered_frames" / (stem + ext)
        if candidate.exists():
            src_img = candidate
            break
    if src_img:
        shutil.copy2(src_img, images_dir / src_img.name)

    label_src = out / "3_labels" / f"{stem}.txt"
    if label_src.exists():
        # Strip optional 6th confidence column — YOLO training needs exactly 5 columns
        raw_lines = label_src.read_text(encoding="utf-8", errors="replace").strip().splitlines()
        clean_lines = [" ".join(l.split()[:5]) for l in raw_lines if l.strip()]
        (labels_dir / f"{stem}.txt").write_text("\n".join(clean_lines), encoding="utf-8")
    ann_src = out / "5_annotated" / filename
    if ann_src.exists():
        shutil.copy2(ann_src, annotated_dir / filename)

    db.mark_saved(job_id, filename)
    return {"ok": True, "image_saved": src_img is not None}


@app.post("/api/skip")
def skip_frame(job_id: str = Form(...), filename: str = Form(...)):
    if not db.get_job(job_id):
        raise HTTPException(404)
    db.mark_skipped(job_id, filename)
    return {"ok": True}


# ── Classes ───────────────────────────────────────────────────────────────────

@app.get("/api/classes")
def list_classes(classes_dir: str = ""):
    c_dir_str = classes_dir.strip() if classes_dir.strip() else db.get_settings()["classes_dir"]
    d = Path(c_dir_str)
    d.mkdir(exist_ok=True)
    return {"classes": sorted(x.name for x in d.iterdir() if x.is_dir())}


@app.post("/api/classes")
def create_class(name: str = Form(...), classes_dir: str = Form(default="")):
    name = name.strip()
    if not name:
        raise HTTPException(400, "Name required")
    c_dir_str = classes_dir.strip() if classes_dir.strip() else db.get_settings()["classes_dir"]
    (Path(c_dir_str) / name).mkdir(parents=True, exist_ok=True)
    return {"ok": True, "name": name}


# ── Models ────────────────────────────────────────────────────────────────────

@app.get("/api/models")
def list_models():
    models = []
    for search_dir in [MODELS_DIR, ROOT]:
        if search_dir.is_dir():
            for f in search_dir.glob("*.pt"):
                models.append({"name": f.name, "path": str(f)})
    return {"models": models}


@app.get("/api/model-classes")
def model_classes(path: str = ""):
    settings   = db.get_settings()
    model_path = Path(path or settings.get("default_yolo_model", ""))
    if not model_path.exists():
        raise HTTPException(404, f"Model not found: {model_path}")
    try:
        from ultralytics import YOLO
        m = YOLO(str(model_path))
        return {"classes": list(m.names.values()), "model": model_path.name, "path": str(model_path)}
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Settings ──────────────────────────────────────────────────────────────────

@app.get("/api/settings")
def get_settings():
    return db.get_settings()


@app.put("/api/settings")
def update_settings(body: dict):
    db.update_settings(body)
    return db.get_settings()


@app.get("/api/browse")
def browse(mode: str = "folder", start: str = "", file_types: str = "", multiselect: bool = False):
    import tkinter as tk
    from tkinter import filedialog
    import threading
    from pathlib import Path
    
    result = {"path": ""}
    event = threading.Event()
    
    def run_dialog():
        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            
            initial = start.strip()
            if initial:
                initial_path = Path(initial)
                if initial_path.exists():
                    if initial_path.is_file():
                        initial = str(initial_path.parent)
                else:
                    initial = ""
            
            if mode == "file":
                filetypes_arg = []
                if file_types:
                    exts = [f"*.{ext.strip().strip('.')}" for ext in file_types.split(",")]
                    filetypes_arg.append(("Supported Files", " ".join(exts)))
                filetypes_arg.append(("All Files", "*.*"))
                
                if multiselect:
                    paths = filedialog.askopenfilenames(
                        initialdir=initial or str(ROOT),
                        title="Select File(s)",
                        filetypes=filetypes_arg
                    )
                    # Join multiple paths with a semicolon
                    result["path"] = ";".join(paths) if paths else ""
                else:
                    path = filedialog.askopenfilename(
                        initialdir=initial or str(ROOT),
                        title="Select File",
                        filetypes=filetypes_arg
                    )
                    result["path"] = path
            else:
                path = filedialog.askdirectory(
                    initialdir=initial or str(ROOT),
                    title="Select Directory"
                )
                result["path"] = path
            root.destroy()
        except Exception as e:
            # Fallback to PowerShell
            import subprocess
            sanitized_start = start.replace("'", "''")
            if mode == "file":
                filter_str = "All Files (*.*)|*.*"
                if file_types:
                    exts = ";".join([f"*.{ext.strip().strip('.')}" for ext in file_types.split(",")])
                    filter_str = f"Supported Files ({exts})|{exts}|All Files (*.*)|*.*"
                
                multiselect_flag = "$d.Multiselect = $true; " if multiselect else ""
                filename_expr = "([string]::Join(';', $d.FileNames))" if multiselect else "$d.FileName"
                
                cmd = (
                    "[System.Reflection.Assembly]::LoadWithPartialName('System.Windows.Forms') | Out-Null; "
                    "$d = New-Object System.Windows.Forms.OpenFileDialog; "
                    "$d.Title = 'Select File(s)'; "
                    f"{multiselect_flag}"
                    f"$d.Filter = '{filter_str}'; "
                    f"$d.InitialDirectory = '{sanitized_start}'; "
                    "$d.ShowDialog() | Out-Null; "
                    f"{filename_expr}"
                )
            else:
                cmd = (
                    "[System.Reflection.Assembly]::LoadWithPartialName('System.Windows.Forms') | Out-Null; "
                    "$d = New-Object System.Windows.Forms.FolderBrowserDialog; "
                    "$d.Description = 'Select Directory'; "
                    f"$d.SelectedPath = '{sanitized_start}'; "
                    "$d.ShowDialog() | Out-Null; "
                    "$d.SelectedPath"
                )
            res = subprocess.run(["powershell", "-Command", cmd], capture_output=True, text=True)
            result["path"] = res.stdout.strip()
        finally:
            event.set()
            
    t = threading.Thread(target=run_dialog)
    t.start()
    event.wait()
    return {"path": result["path"]}


# ── Roboflow ──────────────────────────────────────────────────────────────────

@app.get("/api/roboflow/validate")
def roboflow_validate():
    """Test saved API key + workspace. Returns connection status."""
    settings = db.get_settings()
    api_key   = settings.get("roboflow_api_key", "").strip()
    workspace = settings.get("roboflow_workspace", "").strip()
    if not api_key:
        return {"ok": False, "error": "No Roboflow API key saved in Settings"}
    import roboflow_utils as rf
    return rf.validate_roboflow(api_key, workspace)


@app.get("/api/roboflow/projects")
def roboflow_projects():
    """List projects in the saved workspace."""
    settings = db.get_settings()
    api_key   = settings.get("roboflow_api_key", "").strip()
    workspace = settings.get("roboflow_workspace", "").strip()
    if not api_key or not workspace:
        raise HTTPException(400, "Set Roboflow API key and workspace in Settings first")
    import roboflow_utils as rf
    try:
        return {"projects": rf.list_roboflow_projects(api_key, workspace)}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/roboflow/projects/{project_id}/versions")
def roboflow_versions(project_id: str):
    """List trained model versions for a project."""
    settings = db.get_settings()
    api_key   = settings.get("roboflow_api_key", "").strip()
    workspace = settings.get("roboflow_workspace", "").strip()
    if not api_key or not workspace:
        raise HTTPException(400, "Set Roboflow API key and workspace in Settings first")
    import roboflow_utils as rf
    try:
        return {"versions": rf.get_roboflow_model_versions(api_key, workspace, project_id)}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/roboflow/upload/{job_id}")
def roboflow_upload(job_id: str, project_slug: str = "", batch_name: str = ""):
    """Upload filtered frames from a job to a Roboflow project."""
    j = db.get_job(job_id)
    if not j:
        raise HTTPException(404, "Job not found")
    if not project_slug:
        raise HTTPException(400, "project_slug required")
    settings  = db.get_settings()
    api_key   = settings.get("roboflow_api_key", "").strip()
    workspace = settings.get("roboflow_workspace", "").strip()
    if not api_key:
        raise HTTPException(400, "Set Roboflow API key in Settings first")
    import roboflow_utils as rf
    try:
        result = rf.upload_job_frames(
            job_output_dir=j["output_dir"],
            api_key=api_key,
            workspace=workspace,
            project_slug=project_slug,
            batch_name=batch_name or job_id,
        )
        return result
    except Exception as e:
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
