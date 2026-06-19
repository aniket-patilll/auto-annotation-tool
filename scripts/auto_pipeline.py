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
import sys
from pathlib import Path
from typing import List, Tuple

# Force UTF-8 output on Windows to avoid UnicodeEncodeError with special characters
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

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


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate Structural Similarity Index (SSIM) via OpenCV."""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    k = cv2.getGaussianKernel(11, 1.5)
    w = np.outer(k, k.T)
    f = lambda x: cv2.filter2D(x, -1, w)[5:-5, 5:-5]
    mu1, mu2 = f(a), f(b)
    s1  = f(a*a) - mu1*mu1
    s2  = f(b*b) - mu2*mu2
    s12 = f(a*b) - mu1*mu2
    num = (2*mu1*mu2 + C1) * (2*s12 + C2)
    den = (mu1**2 + mu2**2 + C1) * (s1 + s2 + C2)
    m   = num / (den + 1e-10)
    return float(m.mean())


def frames_similar(a: np.ndarray, b: np.ndarray, threshold: float) -> bool:
    """Checks similarity of two frames using grayscale downsampling and SSIM."""
    sa = cv2.resize(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), (160, 160))
    sb = cv2.resize(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), (160, 160))
    return ssim(sa, sb) >= threshold


def crop_black_bars(frame: np.ndarray, threshold: int = 15, min_fraction: float = 0.5) -> np.ndarray:
    """Crops black pillarboxes/letterboxes from frames."""
    if frame is None:
        return frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    left = 0
    for col in range(w):
        if np.mean(gray[:, col] < threshold) < 0.90:
            left = col
            break
    right = w - 1
    for col in range(w - 1, -1, -1):
        if np.mean(gray[:, col] < threshold) < 0.90:
            right = col
            break

    if (right - left) < int(w * min_fraction):
        return frame
    return frame[:, left:right + 1]


def is_valid_content(image: np.ndarray, threshold: float = 8.0) -> bool:
    """Verify that image standard deviation exceeds blankness threshold."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).std() >= threshold


def filter_frames(
    frame_paths: List[str],
    output_dir: str,
    blur_threshold: float   = 80.0,
    dark_threshold: float   = 25.0,
    bright_threshold: float = 230.0,
    dup_threshold: int      = 8,       # hamming distance; lower = stricter dedup
    dup_method: str         = "hash",  # "hash" or "ssim"
    ssim_threshold: float   = 0.75,
    min_gap: int            = 0,
    min_content_std: float  = 0.0,
    crop_bars: bool         = False,
) -> List[str]:
    """
    Filter out:
      - Near-duplicate frames using hash or SSIM check
      - Blurry frames
      - Bad exposure frames
      - Blank / low content frames
      - Skips frames according to index gaps if min_gap > 0

    Returns list of paths of kept frames (copied or written to output_dir).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    kept = []
    rejected = {"blur": 0, "dark": 0, "bright": 0, "duplicate": 0, "gap": 0, "content": 0}
    last_hash = None
    last_frame = None
    last_idx = -min_gap - 1

    for path in tqdm(frame_paths, desc="  Filtering frames"):
        img = cv2.imread(path)
        if img is None:
            continue

        # Try to parse frame index from name, e.g., name_f000123.jpg
        stem = Path(path).stem
        idx_str = stem.split("_f")[-1]
        frame_idx = int(idx_str) if idx_str.isdigit() else 0

        # Gap filtering
        if min_gap > 0 and (frame_idx - last_idx) < min_gap:
            rejected["gap"] += 1
            continue

        # Black bar cropping
        if crop_bars:
            img = crop_black_bars(img)

        # Standard deviation validation
        if min_content_std > 0.0 and not is_valid_content(img, min_content_std):
            rejected["content"] += 1
            continue

        # Blur validation
        if is_blurry(img, blur_threshold):
            rejected["blur"] += 1
            continue

        # Exposure validations
        if is_too_dark(img, dark_threshold):
            rejected["dark"] += 1
            continue
        if is_too_bright(img, bright_threshold):
            rejected["bright"] += 1
            continue

        # Similarity checks
        if dup_method == "ssim":
            if last_frame is not None and frames_similar(img, last_frame, ssim_threshold):
                rejected["duplicate"] += 1
                continue
        else:
            h = frame_hash(img)
            if last_hash and hamming_distance(h, last_hash) < dup_threshold:
                rejected["duplicate"] += 1
                continue

        # Accept frame
        if dup_method == "ssim":
            last_frame = img
        else:
            last_hash = frame_hash(img)
        last_idx = frame_idx

        dst = output_dir / Path(path).name
        # If cropped, save the modified image. Otherwise, copy for speed.
        if crop_bars:
            cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        else:
            shutil.copy2(path, dst)
        kept.append(str(dst))

    total_in  = len(frame_paths)
    total_out = len(kept)
    pct_kept  = 100 * total_out / total_in if total_in else 0
    print(f"  Filter: {total_in} -> {total_out} frames kept ({pct_kept:.1f}%)")
    print(f"  Rejected — blur:{rejected['blur']}  dark:{rejected['dark']}  "
          f"bright:{rejected['bright']}  dup:{rejected['duplicate']}  "
          f"gap:{rejected['gap']}  content:{rejected['content']}")
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
        print(f"    '{desc}' -> class '{cls}' (idx {class_idx_map.get(cls, '?')})")
    print(f"  Device: {torch_device}")

    model_id = "IDEA-Research/grounding-dino-base"
    print(f"  Loading Grounding DINO ({model_id})...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(torch_device)
    model.eval()

    # Build ". "-separated text prompt covering all class descriptions
    text_prompt = " . ".join(ontology_map.keys()) + " ."

    image_paths = sorted(
        list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.jpeg"))
        + list(image_dir.glob("*.png")) + list(image_dir.glob("*.JPG"))
        + list(image_dir.glob("*.JPEG")) + list(image_dir.glob("*.PNG"))
    )
    stats = {"total": len(image_paths), "labelled": 0, "empty": 0, "error": 0}

    print(f"\n  Auto-labelling {len(image_paths)} images...")

    for img_path in tqdm(image_paths, desc="  Grounding DINO"):
        try:
            # Use PIL directly — avoids cv2.imread issues on some installs
            pil_img = PILImage.open(str(img_path)).convert("RGB")
            W, H = pil_img.size

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
            label_lines = []
            for box, score, raw_label in zip(boxes, scores, labels):
                matched_cls = None
                raw_lower = raw_label.lower()
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
                label_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {score:.4f}")

            label_path = label_dir / img_path.with_suffix(".txt").name
            label_path.write_text("\n".join(label_lines))

            if label_lines:
                stats["labelled"] += 1
            else:
                stats["empty"] += 1

        except Exception as e:
            print(f"\n  [ERR] {img_path.name}: {e}")
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

    image_paths = sorted(
        list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.jpeg"))
        + list(image_dir.glob("*.png")) + list(image_dir.glob("*.JPG"))
        + list(image_dir.glob("*.JPEG")) + list(image_dir.glob("*.PNG"))
    )
    stats = {"total": len(image_paths), "labelled": 0, "empty": 0, "error": 0}

    print(f"\n  Auto-labelling {len(image_paths)} images...")

    for img_path in tqdm(image_paths, desc="  YOLO inference"):
        try:
            results = model.predict(str(img_path), conf=conf_threshold, device=torch_device, verbose=False)

            if not results or results[0].boxes is None or len(results[0].boxes) == 0:
                stats["empty"] += 1
                (label_dir / img_path.with_suffix(".txt").name).write_text("")
                continue

            # Use PIL to get image dimensions — avoids cv2.imread issues
            from PIL import Image as _PILImage
            with _PILImage.open(str(img_path)) as _im:
                W, H = _im.size

            label_lines = []
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                cx = ((x1 + x2) / 2) / W
                cy = ((y1 + y2) / 2) / H
                bw = (x2 - x1) / W
                bh = (y2 - y1) / H
                conf = float(box.conf[0])
                label_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {conf:.4f}")

            (label_dir / img_path.with_suffix(".txt").name).write_text("\n".join(label_lines))
            stats["labelled"] += 1

        except Exception as e:
            print(f"\n  [ERR] {img_path.name}: {e}")
            stats["error"] += 1

    print(f"\n  Auto-label results:")
    print(f"    Labelled with detections : {stats['labelled']}")
    print(f"    Empty (no detections)    : {stats['empty']}")
    print(f"    Errors                   : {stats['error']}")
    return stats, class_names


# ─────────────────────────────────────────────────────────────
# STEP 3c: Auto-labelling with YOLO-World (open-vocabulary)
# ─────────────────────────────────────────────────────────────

def autolabel_yolo_world(
    image_dir: str,
    label_dir: str,
    model_path: str,
    class_names: List[str],
    conf_threshold: float = 0.35,
    device: str = "0",
) -> Tuple[dict, List[str]]:
    """
    YOLO-World with set_classes() for open-vocabulary detection.
    Returns (stats, class_names).
    """
    from ultralytics import YOLO

    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    label_dir.mkdir(parents=True, exist_ok=True)

    torch_device = f"cuda:{device}" if device not in ("cpu", "mps") else device

    print(f"\n  Loading YOLO-World model: {model_path}")
    model = YOLO(model_path)
    model.set_classes(class_names)
    print(f"  Classes ({len(class_names)}): {', '.join(class_names)}")
    print(f"  Device: {torch_device}  |  Conf: {conf_threshold}")

    image_paths = sorted(
        list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.jpeg"))
        + list(image_dir.glob("*.png")) + list(image_dir.glob("*.JPG"))
        + list(image_dir.glob("*.JPEG")) + list(image_dir.glob("*.PNG"))
    )
    stats = {"total": len(image_paths), "labelled": 0, "empty": 0, "error": 0}
    print(f"\n  Auto-labelling {len(image_paths)} images with YOLO-World...")

    for img_path in tqdm(image_paths, desc="  YOLO-World"):
        try:
            results = model.predict(
                str(img_path), conf=conf_threshold, device=torch_device, verbose=False
            )
            if not results or results[0].boxes is None or len(results[0].boxes) == 0:
                stats["empty"] += 1
                (label_dir / img_path.with_suffix(".txt").name).write_text("")
                continue

            # Use PIL for dimensions — avoids cv2.imread issues
            from PIL import Image as _PILImage
            with _PILImage.open(str(img_path)) as _im:
                W, H = _im.size

            label_lines = []
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                cx = ((x1 + x2) / 2) / W
                cy = ((y1 + y2) / 2) / H
                bw = (x2 - x1) / W
                bh = (y2 - y1) / H
                conf = float(box.conf[0])
                label_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {conf:.4f}")

            (label_dir / img_path.with_suffix(".txt").name).write_text(
                "\n".join(label_lines)
            )
            stats["labelled"] += 1
        except Exception as e:
            print(f"\n  [ERR] {img_path.name}: {e}")
            stats["error"] += 1

    print(
        f"\n  YOLO-World results: labelled={stats['labelled']} "
        f"empty={stats['empty']} errors={stats['error']}"
    )
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
    extract_fps: float    = 2.0,
    conf_threshold: float = 0.35,
    train_epochs: int     = 150,
    device: str           = "0",
    model_size: str       = "yolov8m.pt",
    skip_training: bool   = False,
    skip_filter: bool     = False,    # skip blur/dark/dup filtering (use all images as-is)
    labeler: str          = "grounding_dino",
    yolo_model: str       = "",
    input_images: str     = "",   # directory of images — skips frame extraction
    filter_preset: str    = "none",
    dup_method: str       = "hash",
    ssim_threshold: float = 0.75,
    min_frame_gap: int    = 0,
    min_content_std: float = 0.0,
    crop_black_bars: bool  = False,
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

    # ── 1. Extract frames (or use pre-supplied images) ─────────
    print("\n" + "="*55)

    video_files = []    # populated only when processing videos

    if input_images:
        print("STEP 1: Using pre-supplied images (skipping extraction)")
        print("="*55)
        img_dir = Path(input_images)
        all_raw_frames = [
            str(p) for p in sorted(
                list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.jpeg"))
                + list(img_dir.glob("*.png")) + list(img_dir.glob("*.JPG"))
                + list(img_dir.glob("*.JPEG")) + list(img_dir.glob("*.PNG"))
            )
        ]
        print(f"Found {len(all_raw_frames)} images in {input_images}")
        if len(all_raw_frames) == 0:
            # Check for subdirectories
            subdirs = [d for d in img_dir.iterdir() if d.is_dir()]
            if subdirs:
                print(f"  Note: found {len(subdirs)} subdirectory(ies). Images must be directly in the folder, not in subfolders.")
                print(f"  Subdirectories: {[d.name for d in subdirs[:5]]}")
            else:
                all_files = list(img_dir.iterdir())
                print(f"  Folder contains {len(all_files)} file(s) total.")
                if all_files:
                    exts = set(f.suffix.lower() for f in all_files if f.is_file())
                    print(f"  File extensions found: {exts}")
                    print(f"  Supported: .jpg .jpeg .png")
    else:
        print("STEP 1: Extracting frames from videos")
        print("="*55)
        video_extensions = [".mp4", ".mov", ".avi", ".mkv", ".m4v", ".ts"]
        video_path = Path(video_dir)
        if video_path.is_file() and video_path.suffix.lower() in video_extensions:
            video_files = [video_path]
        else:
            video_files = [
                p for p in video_path.glob("**/*")
                if p.suffix.lower() in video_extensions
            ]
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

    if skip_filter or input_images:
        # No filtering — copy images directly to 2_filtered_frames/
        print("STEP 2: Skipping filter — copying all images directly")
        print("="*55)
        import shutil as _shutil
        for fp in all_raw_frames:
            src = Path(fp)
            dst = dirs["filtered_frames"] / src.name
            if not dst.exists():
                _shutil.copy2(src, dst)
        kept_frames = [str(dirs["filtered_frames"] / Path(fp).name) for fp in all_raw_frames]
        print(f"  Copied {len(kept_frames)} images (no filtering applied)")
    else:
        print(f"STEP 2: Filtering frames (blur / dark / duplicates | preset: {filter_preset})")
        print("="*55)

        # Preset threshold resolution
        blur_t = 80.0
        dark_t = 25.0
        bright_t = 230.0
        min_std = min_content_std
        crop_b = crop_black_bars
        dup_m = dup_method
        ssim_t = ssim_threshold
        gap = min_frame_gap

        if filter_preset == "handheld":
            dup_m = "ssim"
            ssim_t = 0.75
            gap = 3
            blur_t = 80.0
            dark_t = 30.0
            bright_t = 230.0
            min_std = 8.0
            crop_b = True
        elif filter_preset == "cctv":
            dup_m = "ssim"
            ssim_t = 0.98
            gap = 10
            blur_t = 40.0
            dark_t = 20.0
            bright_t = 240.0
            min_std = 8.0
            crop_b = True

        kept_frames = filter_frames(
            all_raw_frames,
            str(dirs["filtered_frames"]),
            blur_threshold=blur_t,
            dark_threshold=dark_t,
            bright_threshold=bright_t,
            dup_method=dup_m,
            ssim_threshold=ssim_t,
            min_gap=gap,
            min_content_std=min_std,
            crop_bars=crop_b,
        )
        print(f"Frames after filtering: {len(kept_frames)}")

    # ── 3. Auto-label ──────────────────────────────────────────
    print("\n" + "="*55)
    _labeler_names = {
        "yolo": "trained YOLO",
        "yolo_world": "YOLO-World",
        "grounding_dino": "Grounding DINO",
    }
    print(f"STEP 3: Auto-labelling with {_labeler_names.get(labeler, labeler)}")
    print("="*55)

    if labeler == "yolo":
        if not yolo_model:
            raise ValueError("--yolo-model required when --labeler yolo")
        label_stats, class_names = autolabel_yolo(
            str(dirs["filtered_frames"]), str(dirs["labels"]),
            model_path=yolo_model, conf_threshold=conf_threshold, device=device,
        )
    elif labeler == "yolo_world":
        if not yolo_model:
            raise ValueError("--yolo-model required when --labeler yolo_world")
        label_stats, class_names = autolabel_yolo_world(
            str(dirs["filtered_frames"]), str(dirs["labels"]),
            model_path=yolo_model, class_names=class_names,
            conf_threshold=conf_threshold, device=device,
        )
    else:
        label_stats = autolabel_images(
            str(dirs["filtered_frames"]), str(dirs["labels"]),
            class_names, conf_threshold=conf_threshold, device=device,
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
        description="Annotation pipeline: video/images → auto-label → YOLO dataset"
    )
    parser.add_argument("--videos",        default="",
        help="Folder of video files (mp4/mov/avi/mkv). Required unless --input-images given.")
    parser.add_argument("--input-images",  default="",
        help="Folder of images to label directly (skips frame extraction).")
    parser.add_argument("--output",        default="./pipeline_output",
        help="Output folder for frames, labels, dataset")
    parser.add_argument("--classes",       required=True,
        help="Comma-separated class names, e.g. burger,fries,drink")
    parser.add_argument("--fps",           type=float, default=1.0)
    parser.add_argument("--conf",          type=float, default=0.35)
    parser.add_argument("--epochs",        type=int,   default=150)
    parser.add_argument("--device",        default="0")
    parser.add_argument("--model-size",    default="yolov8m.pt")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-filter",   action="store_true",
        help="Skip blur/dark/duplicate filtering — use all images as-is (recommended for pre-curated image sets)")
    parser.add_argument("--labeler",       default="grounding_dino",
        choices=["grounding_dino", "yolo", "yolo_world"],
        help="Auto-labeller: grounding_dino | yolo | yolo_world")
    parser.add_argument("--yolo-model",    default="",
        help="Path to .pt when --labeler yolo or yolo_world")
    parser.add_argument("--filter-preset", default="none", choices=["none", "handheld", "cctv"],
        help="Filter configuration preset. If set to 'handheld' or 'cctv', automatically configures SSIM duplication checks, exposure limits, gap constraints, and black bar cropping.")
    parser.add_argument("--dup-method",    default="hash", choices=["hash", "ssim"],
        help="Duplicate detection method: 'hash' (fast perceptual hash) or 'ssim' (accurate structural similarity)")
    parser.add_argument("--ssim-threshold", type=float, default=0.75,
        help="SSIM threshold for duplicate detection (used when --dup-method ssim)")
    parser.add_argument("--min-frame-gap",  type=int, default=0,
        help="Minimum frame index gap between kept frames")
    parser.add_argument("--min-content-std", type=float, default=0.0,
        help="Minimum pixel standard deviation to filter blank frames")
    parser.add_argument("--crop-black-bars", action="store_true",
        help="Crop black bar borders from frame edges")

    args = parser.parse_args()
    if not args.videos and not args.input_images:
        parser.error("Provide --videos or --input-images")

    run_pipeline(
        video_dir      = args.videos,
        output_dir     = args.output,
        class_names    = [c.strip() for c in args.classes.split(",")],
        extract_fps    = args.fps,
        conf_threshold = args.conf,
        train_epochs   = args.epochs,
        device         = args.device,
        model_size     = args.model_size,
        skip_training  = args.skip_training,
        skip_filter    = args.skip_filter,
        labeler        = args.labeler,
        yolo_model     = args.yolo_model,
        input_images   = args.input_images,
        filter_preset  = args.filter_preset,
        dup_method     = args.dup_method,
        ssim_threshold = args.ssim_threshold,
        min_frame_gap  = args.min_frame_gap,
        min_content_std = args.min_content_std,
        crop_black_bars = args.crop_black_bars,
    )
