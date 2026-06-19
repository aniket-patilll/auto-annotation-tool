"""
Visualize Grounding DINO auto-labels on extracted frames.

Usage:
    # Single video output:
    python visualize_labels.py --pipeline-output ./pipeline_output/test1

    # All videos at once:
    python visualize_labels.py --pipeline-output ./pipeline_output --all

Saves annotated images to <pipeline_output>/<video>/5_annotated/
"""

import argparse
import cv2
import numpy as np
from pathlib import Path

COLORS = [
    (0, 255, 0),    # burger    - green
    (0, 165, 255),  # fries     - orange
    (255, 0, 0),    # drink     - blue
    (0, 255, 255),  # nuggets   - yellow
    (255, 0, 255),  # wrap      - magenta
    (255, 128, 0),  # sandwich  - sky blue
    (128, 0, 255),  # bag       - purple
]


def draw_labels(image_path: Path, label_path: Path, class_names: list) -> np.ndarray:
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    H, W = img.shape[:2]

    if not label_path.exists() or label_path.stat().st_size == 0:
        # No detections — mark frame
        cv2.putText(img, "NO DETECTIONS", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return img

    for line in label_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls_id, cx, cy, bw, bh = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

        x1 = int((cx - bw / 2) * W)
        y1 = int((cy - bh / 2) * H)
        x2 = int((cx + bw / 2) * W)
        y2 = int((cy + bh / 2) * H)

        color = COLORS[cls_id % len(COLORS)]
        label = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(img, label, (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return img


def visualize_one(output_dir: Path):
    frames_dir = output_dir / "2_filtered_frames"
    labels_dir = output_dir / "3_labels"
    annotated_dir = output_dir / "5_annotated"

    if not frames_dir.exists():
        print(f"  ✗ No filtered frames at {frames_dir}")
        return

    # Read class names from data.yaml
    yaml_path = output_dir / "4_dataset" / "data.yaml"
    class_names = ["burger", "fries", "drink", "nuggets", "wrap", "sandwich", "bag"]
    if yaml_path.exists():
        for line in yaml_path.read_text().splitlines():
            if line.strip().startswith("names:"):
                raw = line.split("names:")[1].strip()
                class_names = [c.strip().strip("'\"") for c in raw.strip("[]").split(",")]
                break

    annotated_dir.mkdir(exist_ok=True)
    image_paths = sorted(list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png")))

    saved = 0
    for img_path in image_paths:
        label_path = labels_dir / img_path.with_suffix(".txt").name
        annotated = draw_labels(img_path, label_path, class_names)
        if annotated is not None:
            out_path = annotated_dir / img_path.name
            cv2.imwrite(str(out_path), annotated)
            saved += 1

    print(f"  ✓ {output_dir.name}: {saved} annotated images → {annotated_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline-output", required=True,
                        help="Path to a single video output dir OR parent dir when --all")
    parser.add_argument("--all", action="store_true",
                        help="Process all subdirs under --pipeline-output")
    args = parser.parse_args()

    root = Path(args.pipeline_output)

    if args.all:
        subdirs = [d for d in sorted(root.iterdir()) if d.is_dir() and (d / "2_filtered_frames").exists()]
        print(f"Found {len(subdirs)} video output(s) under {root}\n")
        for d in subdirs:
            visualize_one(d)
    else:
        visualize_one(root)

    print("\nDone. Open 5_annotated/ folder in any image viewer to inspect labels.")
