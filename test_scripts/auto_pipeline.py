"""
KitchEye — Fully Automated Video → YOLO Training Pipeline
==========================================================
Usage:
    python auto_pipeline.py --videos ./videos --output ./dataset --classes burger,fries,drink,nuggets,wrap,salad,dip

Steps:
    1. Extract frames from all video files
    2. Filter: remove blurry, blank, and near-duplicate frames
    3. Auto-label with Grounding DINO (HuggingFace transformers)
    4. Filter by confidence — keep strong labels, discard weak
    5. Write YOLO-format dataset (images + labels + data.yaml)
    6. Train YOLOv8 on the dataset

Install:
    pip install opencv-python torch torchvision ultralytics
    pip install transformers Pillow tqdm
"""

import argparse
import os
import shutil
import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────
# STEP 1: Frame Extraction
# ─────────────────────────────────────────────────────────────

def extract_frames(
    video_path: str,
    output_dir: str,
    fps: float = 2.0,          # extract N frames per second
    max_frames: int = 5000,     # safety cap per video
) -> List[str]:
    """
    Extract frames from a video at a given FPS rate.
    Returns list of saved frame paths.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ✗ Cannot open: {video_path}")
        return []

    video_fps   = cap.get(cv2.CAP_PROP_FPS) or 25
    total       = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval    = max(1, int(video_fps / fps))   # read every Nth frame
    stem        = video_path.stem

    saved = []
    frame_idx = 0
    extracted = 0

    print(f"  Video: {video_path.name}  |  {total} frames @ {video_fps:.1f} fps")
    print(f"  Extracting every {interval} frames ({fps} fps target)")

    while cap.isOpened() and extracted < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            fname = output_dir / f"{stem}_f{frame_idx:06d}.jpg"
            cv2.imwrite(str(fname), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            saved.append(str(fname))
            extracted += 1
        frame_idx += 1

    cap.release()
    print(f"  Extracted: {extracted} frames")
    return saved


# ─────────────────────────────────────────────────────────────
# STEP 2: Frame Quality Filtering
# ─────────────────────────────────────────────────────────────

def is_blurry(image: np.ndarray, threshold: float = 80.0) -> bool:
    """Laplacian variance — low = blurry."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold


def is_too_dark(image: np.ndarray, threshold: float = 25.0) -> bool:
    """Mean brightness — reject very dark frames."""
    return image.mean() < threshold


def is_too_bright(image: np.ndarray, threshold: float = 230.0) -> bool:
    """Reject overexposed frames."""
    return image.mean() > threshold


def frame_hash(image: np.ndarray, size: int = 16) -> str:
    """
    Perceptual hash for near-duplicate detection.
    Resize to small greyscale grid, return binary string.
    """
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (size, size))
    avg   = small.mean()
    bits  = (small > avg).flatten()
    return ''.join('1' if b else '0' for b in bits)


def hamming_distance(h1: str, h2: str) -> int:
    return sum(c1 != c2 for c1, c2 in zip(h1, h2))


def filter_frames(
    frame_paths: List[str],
    output_dir: str,
    blur_threshold: float   = 80.0,
    dark_threshold: float   = 25.0,
    bright_threshold: float = 230.0,
    dup_threshold: int      = 8,       # hamming distance; lower = stricter dedup
) -> List[str]:
    """
    Filter out:
      - Blurry frames
      - Too dark / too bright frames
      - Near-duplicate frames (similar to the previous kept frame)

    Returns list of paths of kept frames (copied to output_dir).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    kept = []
    rejected = {"blur": 0, "dark": 0, "bright": 0, "duplicate": 0}
    last_hash = None

    for path in tqdm(frame_paths, desc="  Filtering frames"):
        img = cv2.imread(path)
        if img is None:
            continue

        if is_blurry(img, blur_threshold):
            rejected["blur"] += 1
            continue
        if is_too_dark(img, dark_threshold):
            rejected["dark"] += 1
            continue
        if is_too_bright(img, bright_threshold):
            rejected["bright"] += 1
            continue

        h = frame_hash(img)
        if last_hash and hamming_distance(h, last_hash) < dup_threshold:
            rejected["duplicate"] += 1
            continue

        last_hash = h
        dst = output_dir / Path(path).name
        shutil.copy2(path, dst)
        kept.append(str(dst))

    total_in  = len(frame_paths)
    total_out = len(kept)
    pct_kept  = 100 * total_out / total_in if total_in else 0
    print(f"  Filter: {total_in} -> {total_out} frames kept ({pct_kept:.1f}%)")
    print(f"  Rejected — blur:{rejected['blur']}  dark:{rejected['dark']}  "
          f"bright:{rejected['bright']}  dup:{rejected['duplicate']}")
    return kept


# ─────────────────────────────────────────────────────────────
# STEP 3: Auto-labelling with Grounding DINO
# ─────────────────────────────────────────────────────────────

def build_ontology(class_names: List[str]) -> dict:
    """
    Build a text→class mapping for Grounding DINO.
    Adds natural language variations for common kitchen items.
    """
    synonyms = {
        "burger":         "hamburger burger beef patty sandwich",
        "fries":          "french fries potato fries chips",
        "drink":          "drink cup beverage cup soda cup",
        "nuggets":        "chicken nuggets nuggets pieces",
        "wrap":           "chicken wrap tortilla wrap",
        "salad":          "salad bowl garden salad",
        "dip":            "dipping sauce sauce cup dip pot",
        "sandwich":       "chicken sandwich sandwich",
        "bag":            "paper bag food bag packaging",
        "cheese":         "cheese slice cheese",
        "dessert":        "dessert cake pastry",
    }
    ontology = {}
    for cls in class_names:
        key = cls.lower().strip()
        # Use multi-word description if available, else the class name itself
        description = synonyms.get(key, key.replace("_", " "))
        ontology[description] = key
    return ontology


def autolabel_images(
    image_dir: str,
    label_dir: str,
    class_names: List[str],
    conf_threshold: float = 0.35,
    device: str = "0",
) -> dict:
    """
    Run Grounding DINO (HuggingFace) on all images and save YOLO-format labels.
    Returns stats: {total, labelled, empty, error}
    """
    import torch
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    from PIL import Image as PILImage

    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    label_dir.mkdir(parents=True, exist_ok=True)

    torch_device = (
        f"cuda:{device}" if device not in ("cpu", "mps") and torch.cuda.is_available()
        else device if device == "mps"
        else "cpu"
    )

    ontology_map  = build_ontology(class_names)
    class_idx_map = {name: i for i, name in enumerate(class_names)}

    print(f"\n  Ontology:")
    for desc, cls in ontology_map.items():
        print(f"    '{desc}' → class '{cls}' (idx {class_idx_map.get(cls, '?')})")
    print(f"  Device: {torch_device}")

    model_id = "IDEA-Research/grounding-dino-base"
    print(f"  Loading Grounding DINO ({model_id})...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(torch_device)
    model.eval()

    # Build ". "-separated text prompt covering all class descriptions
    text_prompt = " . ".join(ontology_map.keys()) + " ."

    image_paths = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))
    stats = {"total": len(image_paths), "labelled": 0, "empty": 0, "error": 0}

    print(f"\n  Auto-labelling {len(image_paths)} images...")

    for img_path in tqdm(image_paths, desc="  Grounding DINO"):
        try:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                stats["error"] += 1
                continue

            H, W = img_bgr.shape[:2]
            pil_img = PILImage.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

            inputs = processor(images=pil_img, text=text_prompt, return_tensors="pt")
            inputs = {k: v.to(torch_device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                box_threshold=conf_threshold,
                text_threshold=conf_threshold,
                target_sizes=[(H, W)],
            )[0]

            boxes  = results["boxes"].tolist()
            scores = results["scores"].tolist()
            labels = results["labels"]  # text strings from DINO

            # Map DINO text labels back to class indices
            # Match each returned label to best-matching class via ontology
            label_lines = []
            for box, score, raw_label in zip(boxes, scores, labels):
                matched_cls = None
                raw_lower = raw_label.lower()
                # Find which ontology description this label came from
                for desc, cls_name in ontology_map.items():
                    if any(word in raw_lower for word in desc.split()):
                        matched_cls = cls_name
                        break
                if matched_cls is None:
                    matched_cls = class_names[0]  # fallback

                cls_id = class_idx_map.get(matched_cls, 0)
                x1, y1, x2, y2 = box
                cx = ((x1 + x2) / 2) / W
                cy = ((y1 + y2) / 2) / H
                bw = (x2 - x1) / W
                bh = (y2 - y1) / H
                label_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            label_path = label_dir / img_path.with_suffix(".txt").name
            label_path.write_text("\n".join(label_lines))

            if label_lines:
                stats["labelled"] += 1
            else:
                stats["empty"] += 1

        except Exception as e:
            print(f"\n  ✗ Error on {img_path.name}: {e}")
            stats["error"] += 1

    print(f"\n  Auto-label results:")
    print(f"    Labelled with detections : {stats['labelled']}")
    print(f"    Empty (no detections)    : {stats['empty']}")
    print(f"    Errors                   : {stats['error']}")
    return stats


# ─────────────────────────────────────────────────────────────
# STEP 3b: Auto-labelling with trained YOLO model
# ─────────────────────────────────────────────────────────────

def autolabel_yolo(
    image_dir: str,
    label_dir: str,
    model_path: str,
    conf_threshold: float = 0.35,
    device: str = "0",
) -> Tuple[dict, List[str]]:
    """
    Run a trained YOLOv8 model on all images and save YOLO-format labels.
    Class IDs and names come directly from the model — no ontology mapping needed.
    Returns (stats, class_names).
    """
    from ultralytics import YOLO

    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    label_dir.mkdir(parents=True, exist_ok=True)

    torch_device = (
        f"cuda:{device}" if device not in ("cpu", "mps") else device
    )

    print(f"\n  Loading YOLO model: {model_path}")
    model = YOLO(model_path)
    class_names = list(model.names.values())
    print(f"  Classes ({len(class_names)}): {', '.join(class_names)}")
    print(f"  Device: {torch_device}  |  Conf threshold: {conf_threshold}")

    image_paths = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))
    stats = {"total": len(image_paths), "labelled": 0, "empty": 0, "error": 0}

    print(f"\n  Auto-labelling {len(image_paths)} images...")

    for img_path in tqdm(image_paths, desc="  YOLO inference"):
        try:
            results = model.predict(str(img_path), conf=conf_threshold, device=torch_device, verbose=False)

            if not results or results[0].boxes is None or len(results[0].boxes) == 0:
                stats["empty"] += 1
                (label_dir / img_path.with_suffix(".txt").name).write_text("")
                continue

            img = cv2.imread(str(img_path))
            H, W = img.shape[:2]

            label_lines = []
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                cx = ((x1 + x2) / 2) / W
                cy = ((y1 + y2) / 2) / H
                bw = (x2 - x1) / W
                bh = (y2 - y1) / H
                label_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            (label_dir / img_path.with_suffix(".txt").name).write_text("\n".join(label_lines))
            stats["labelled"] += 1

        except Exception as e:
            print(f"\n  ✗ Error on {img_path.name}: {e}")
            stats["error"] += 1

    print(f"\n  Auto-label results:")
    print(f"    Labelled with detections : {stats['labelled']}")
    print(f"    Empty (no detections)    : {stats['empty']}")
    print(f"    Errors                   : {stats['error']}")
    return stats, class_names


# ─────────────────────────────────────────────────────────────
# STEP 4: Write YOLO dataset structure
# ─────────────────────────────────────────────────────────────

def write_yolo_dataset(
    image_dir: str,
    label_dir: str,
    output_dir: str,
    class_names: List[str],
    val_split: float = 0.15,
) -> str:
    """
    Organise images + labels into YOLO train/val split.
    Returns path to data.yaml.
    """
    import random

    output_dir = Path(output_dir)
    image_dir  = Path(image_dir)
    label_dir  = Path(label_dir)

    train_img = output_dir / "images" / "train"
    val_img   = output_dir / "images" / "val"
    train_lbl = output_dir / "labels" / "train"
    val_lbl   = output_dir / "labels" / "val"
    for d in [train_img, val_img, train_lbl, val_lbl]:
        d.mkdir(parents=True, exist_ok=True)

    # Only include images that have a corresponding label file
    image_files = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))
    pairs = []
    for img in image_files:
        lbl = label_dir / img.with_suffix(".txt").name
        if lbl.exists():
            pairs.append((img, lbl))

    random.shuffle(pairs)
    split_idx = max(1, int(len(pairs) * val_split))
    val_pairs   = pairs[:split_idx]
    train_pairs = pairs[split_idx:]

    def copy_pairs(pair_list, img_dst, lbl_dst):
        for img, lbl in pair_list:
            shutil.copy2(img, img_dst / img.name)
            shutil.copy2(lbl, lbl_dst / lbl.name)

    copy_pairs(train_pairs, train_img, train_lbl)
    copy_pairs(val_pairs,   val_img,   val_lbl)

    # Write data.yaml
    yaml_path = output_dir / "data.yaml"
    yaml_content = f"""# KitchEye auto-generated dataset
path: {output_dir.resolve()}
train: images/train
val:   images/val

nc: {len(class_names)}
names: {class_names}
"""
    yaml_path.write_text(yaml_content)

    print(f"\n  Dataset written:")
    print(f"    Train: {len(train_pairs)} images")
    print(f"    Val:   {len(val_pairs)} images")
    print(f"    Classes ({len(class_names)}): {', '.join(class_names)}")
    print(f"    data.yaml: {yaml_path}")
    return str(yaml_path)


# ─────────────────────────────────────────────────────────────
# STEP 5: Train YOLOv8
# ─────────────────────────────────────────────────────────────

def train_yolo(
    data_yaml: str,
    model_size: str  = "yolov8m.pt",   # n=nano, s=small, m=medium — m fits RTX 3050 4GB
    epochs: int      = 150,
    imgsz: int       = 640,
    batch: int       = 16,
    output_dir: str  = "./runs",
    device: str      = "0",            # "0" for GPU, "cpu", "mps"
) -> str:
    """Train YOLOv8 on the auto-labelled dataset."""
    from ultralytics import YOLO

    print(f"\n  Training YOLOv8 ({model_size})")
    print(f"  Epochs: {epochs}  |  Image size: {imgsz}  |  Batch: {batch}")

    model = YOLO(model_size)
    results = model.train(
        data    = data_yaml,
        epochs  = epochs,
        imgsz   = imgsz,
        batch   = batch,
        device  = device,
        project = output_dir,
        name    = "kitcheye_auto",
        exist_ok= True,
        # Augmentation to compensate for overhead CCTV domain shift
        degrees  = 45,      # random rotation ±45°
        flipud   = 0.5,     # vertical flip (overhead views look same flipped)
        fliplr   = 0.5,
        hsv_h    = 0.02,
        hsv_s    = 0.5,
        hsv_v    = 0.4,
        mosaic   = 1.0,
        mixup    = 0.1,
    )

    best_model = Path(output_dir) / "kitcheye_auto" / "weights" / "best.pt"
    print(f"\n  ✓ Training complete. Best model: {best_model}")
    return str(best_model)


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────

def run_pipeline(
    video_dir: str,
    output_dir: str,
    class_names: List[str],
    extract_fps: float  = 2.0,
    conf_threshold: float = 0.35,
    train_epochs: int   = 150,
    device: str         = "0",
    model_size: str     = "yolov8m.pt",
    skip_training: bool = False,
    labeler: str        = "grounding_dino",   # "grounding_dino" or "yolo"
    yolo_model: str     = "",                 # path to .pt when labeler="yolo"
):
    output_dir = Path(output_dir)
    dirs = {
        "raw_frames":      output_dir / "1_raw_frames",
        "filtered_frames": output_dir / "2_filtered_frames",
        "labels":          output_dir / "3_labels",
        "dataset":         output_dir / "4_dataset",
        "model":           output_dir / "5_model",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # ── 1. Extract frames ──────────────────────────────────────
    print("\n" + "="*55)
    print("STEP 1: Extracting frames from videos")
    print("="*55)

    video_extensions = [".mp4", ".mov", ".avi", ".mkv", ".m4v", ".ts"]
    video_path = Path(video_dir)
    if video_path.is_file() and video_path.suffix.lower() in video_extensions:
        video_files = [video_path]
    else:
        video_files = [p for p in video_path.glob("**/*") if p.suffix.lower() in video_extensions]
    print(f"Found {len(video_files)} video file(s) in {video_dir}")

    all_raw_frames = []
    for vf in video_files:
        frames = extract_frames(
            str(vf),
            str(dirs["raw_frames"] / vf.stem),
            fps=extract_fps,
        )
        all_raw_frames.extend(frames)

    print(f"\nTotal raw frames extracted: {len(all_raw_frames)}")

    # ── 2. Filter frames ────────────────────────────────────────
    print("\n" + "="*55)
    print("STEP 2: Filtering frames (blur / dark / duplicates)")
    print("="*55)

    kept_frames = filter_frames(
        all_raw_frames,
        str(dirs["filtered_frames"]),
    )
    print(f"Frames after filtering: {len(kept_frames)}")

    # ── 3. Auto-label ──────────────────────────────────────────
    print("\n" + "="*55)
    if labeler == "yolo":
        print("STEP 3: Auto-labelling with trained YOLO model")
    else:
        print("STEP 3: Auto-labelling with Grounding DINO")
    print("="*55)

    if labeler == "yolo":
        if not yolo_model:
            raise ValueError("--yolo-model path required when --labeler yolo")
        label_stats, class_names = autolabel_yolo(
            str(dirs["filtered_frames"]),
            str(dirs["labels"]),
            model_path=yolo_model,
            conf_threshold=conf_threshold,
            device=device,
        )
    else:
        label_stats = autolabel_images(
            str(dirs["filtered_frames"]),
            str(dirs["labels"]),
            class_names,
            conf_threshold=conf_threshold,
            device=device,
        )

    # ── 4. Write dataset ────────────────────────────────────────
    print("\n" + "="*55)
    print("STEP 4: Writing YOLO dataset")
    print("="*55)

    data_yaml = write_yolo_dataset(
        str(dirs["filtered_frames"]),
        str(dirs["labels"]),
        str(dirs["dataset"]),
        class_names,
    )

    # Save pipeline run summary
    summary = {
        "videos_processed":  len(video_files),
        "raw_frames":        len(all_raw_frames),
        "filtered_frames":   len(kept_frames),
        "labelled_frames":   label_stats["labelled"],
        "empty_frames":      label_stats["empty"],
        "class_names":       class_names,
        "conf_threshold":    conf_threshold,
        "data_yaml":         data_yaml,
    }
    (output_dir / "pipeline_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(f"\n  Pipeline summary saved: {output_dir / 'pipeline_summary.json'}")

    # ── 5. Train ───────────────────────────────────────────────
    if skip_training:
        print("\n⏭  Skipping training (--skip-training flag set)")
        print(f"  Dataset ready at: {data_yaml}")
        print("  Run training manually:")
        print(f"  yolo train data={data_yaml} model=yolov8s.pt epochs=150")
        return data_yaml

    print("\n" + "="*55)
    print("STEP 5: Training YOLOv8")
    print("="*55)

    best_model = train_yolo(
        data_yaml    = data_yaml,
        model_size   = model_size,
        epochs       = train_epochs,
        output_dir   = str(dirs["model"]),
        device       = device,
    )

    print("\n" + "="*55)
    print("✓  PIPELINE COMPLETE")
    print("="*55)
    print(f"  Videos processed : {len(video_files)}")
    print(f"  Frames extracted : {len(all_raw_frames)}")
    print(f"  Frames kept      : {len(kept_frames)}")
    print(f"  Frames labelled  : {label_stats['labelled']}")
    print(f"  Best model       : {best_model}")
    print("="*55)
    return best_model


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="KitchEye: Video → auto-labelled YOLO training pipeline"
    )
    parser.add_argument("--videos",   required=True,
        help="Folder containing video files (mp4, mov, avi, mkv)")
    parser.add_argument("--output",   default="./pipeline_output",
        help="Output folder for frames, labels, dataset, model")
    parser.add_argument("--classes",  required=True,
        help="Comma-separated class names, e.g. burger,fries,drink,nuggets")
    parser.add_argument("--fps",      type=float, default=2.0,
        help="Frames per second to extract from each video (default: 2)")
    parser.add_argument("--conf",     type=float, default=0.35,
        help="Minimum Grounding DINO confidence to keep a label (default: 0.35)")
    parser.add_argument("--epochs",   type=int, default=150,
        help="YOLOv8 training epochs (default: 150)")
    parser.add_argument("--device",   default="0",
        help="Device: '0' for GPU, 'cpu', 'mps' for Apple Silicon")
    parser.add_argument("--model-size", default="yolov8m.pt",
        help="YOLOv8 model for training: yolov8n.pt / yolov8s.pt / yolov8m.pt (default: yolov8m.pt)")
    parser.add_argument("--skip-training", action="store_true",
        help="Stop after generating dataset, skip YOLO training")
    parser.add_argument("--labeler", default="grounding_dino",
        choices=["grounding_dino", "yolo"],
        help="Auto-labeller to use: grounding_dino (default) or yolo")
    parser.add_argument("--yolo-model", default="",
        help="Path to trained .pt model when --labeler yolo")

    args = parser.parse_args()
    class_names = [c.strip() for c in args.classes.split(",")]

    run_pipeline(
        video_dir     = args.videos,
        output_dir    = args.output,
        class_names   = class_names,
        extract_fps   = args.fps,
        conf_threshold= args.conf,
        train_epochs  = args.epochs,
        device        = args.device,
        model_size    = args.model_size,
        skip_training = args.skip_training,
        labeler       = args.labeler,
        yolo_model    = args.yolo_model,
    )
