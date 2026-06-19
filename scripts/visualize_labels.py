"""
Visualize auto-labels on extracted frames.
Uses Pillow only — no cv2 dependency (avoids NumPy 1/2 compatibility issues).

Usage:
    python visualize_labels.py --pipeline-output ./pipeline_output/test1
    python visualize_labels.py --pipeline-output ./pipeline_output --all
"""

import sys
import argparse
from pathlib import Path

# Force UTF-8 output on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from PIL import Image, ImageDraw, ImageFont

# Class colours (R, G, B)
COLORS = [
    (76,  175,  80),   # green
    (255, 152,   0),   # orange
    ( 33, 150, 243),   # blue
    (244,  67,  54),   # red
    (156,  39, 176),   # purple
    (  0, 188, 212),   # cyan
    (255, 235,  59),   # yellow
    (233,  30,  99),   # pink
    ( 96, 125, 139),   # blue-grey
]


def _try_font(size: int = 16):
    """Try to load a truetype font; fall back to PIL default."""
    font_candidates = [
        "arial.ttf", "Arial.ttf",
        "DejaVuSans.ttf", "DejaVuSans-Bold.ttf",
        "LiberationSans-Regular.ttf",
    ]
    for name in font_candidates:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def draw_labels(image_path: Path, label_path: Path, class_names: list):
    """Draw bounding boxes on image. Returns PIL Image or None on failure."""
    try:
        img = Image.open(str(image_path)).convert("RGB")
    except Exception as e:
        print(f"  [ERR] Cannot open {image_path.name}: {e}")
        return None

    W, H = img.size
    draw = ImageDraw.Draw(img, "RGBA")
    font = _try_font(max(14, H // 40))

    if not label_path.exists() or label_path.stat().st_size == 0:
        # Mark as no detections
        draw.rectangle([(0, 0), (W, 30)], fill=(200, 0, 0, 180))
        draw.text((4, 4), "NO DETECTIONS", fill=(255, 255, 255), font=font)
        return img

    for line in label_path.read_text(encoding="utf-8", errors="replace").strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        # 6th column is optional confidence score (0.0–1.0)
        conf = float(parts[5]) if len(parts) >= 6 else None

        x1 = int((cx - bw / 2) * W)
        y1 = int((cy - bh / 2) * H)
        x2 = int((cx + bw / 2) * W)
        y2 = int((cy + bh / 2) * H)

        color = COLORS[cls_id % len(COLORS)]
        class_label = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
        # Build display label: "drums 97%" or just "drums"
        label = f"{class_label} {conf*100:.0f}%" if conf is not None else class_label

        # Box outline (3px)
        for t in range(3):
            draw.rectangle([(x1 - t, y1 - t), (x2 + t, y2 + t)], outline=color)

        # Label badge
        try:
            bbox = font.getbbox(label)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
        except Exception:
            tw, th = len(label) * 8, 16

        lx = x1
        ly = max(0, y1 - th - 6)
        draw.rectangle([(lx, ly), (lx + tw + 8, ly + th + 6)], fill=color + (220,))
        draw.text((lx + 4, ly + 3), label, fill=(255, 255, 255), font=font)

    return img



def _all_images(folder: Path) -> list:
    """Return all image paths from folder, sorted."""
    exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG", "*.bmp", "*.BMP"]
    found = []
    for ext in exts:
        found.extend(folder.glob(ext))
    return sorted(set(found), key=lambda p: p.name)


def visualize_one(output_dir: Path):
    frames_dir    = output_dir / "2_filtered_frames"
    labels_dir    = output_dir / "3_labels"
    annotated_dir = output_dir / "5_annotated"

    if not frames_dir.exists():
        print(f"  [SKIP] No filtered frames at {frames_dir}")
        return

    # Read class names from data.yaml
    yaml_path = output_dir / "4_dataset" / "data.yaml"
    class_names = ["object"]
    if yaml_path.exists():
        for line in yaml_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.strip().startswith("names:"):
                raw = line.split("names:")[1].strip()
                class_names = [c.strip().strip("'\"") for c in raw.strip("[]").split(",")]
                break

    annotated_dir.mkdir(exist_ok=True)
    image_paths = _all_images(frames_dir)

    saved = 0
    errors = 0
    for img_path in image_paths:
        label_path = labels_dir / img_path.with_suffix(".txt").name
        result = draw_labels(img_path, label_path, class_names)
        if result is not None:
            out_path = annotated_dir / (img_path.stem + ".jpg")
            result.save(str(out_path), "JPEG", quality=92)
            saved += 1
        else:
            errors += 1

    print(f"  [OK] {output_dir.name}: {saved}/{len(image_paths)} annotated images -> {annotated_dir}")
    if errors:
        print(f"  [WARN] {errors} images could not be processed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline-output", required=True,
                        help="Path to a pipeline output dir OR parent dir when --all")
    parser.add_argument("--all", action="store_true",
                        help="Process all subdirs under --pipeline-output")
    args = parser.parse_args()

    root = Path(args.pipeline_output)

    if args.all:
        subdirs = [d for d in sorted(root.iterdir())
                   if d.is_dir() and (d / "2_filtered_frames").exists()]
        print(f"Found {len(subdirs)} pipeline output(s) under {root}\n")
        for d in subdirs:
            visualize_one(d)
    else:
        visualize_one(root)

    print("\nDone. Open 5_annotated/ to inspect labels.")
