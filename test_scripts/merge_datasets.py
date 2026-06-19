"""
KitchEye — Merge Per-Video YOLO Datasets into One Combined Dataset
==================================================================
Scans pipeline_output/*/4_dataset/ and combines all images + labels
into a single train/val split for YOLOv8 training.

Usage:
    python test_scripts/merge_datasets.py \
        --pipeline-dir ./pipeline_output \
        --output ./pipeline_output/combined_dataset
"""

import argparse
import shutil
import random
from pathlib import Path


CLASS_NAMES = ["burger", "fries", "drink", "nuggets", "wrap", "sandwich", "bag"]


def merge_datasets(pipeline_dir: str, output_dir: str, val_split: float = 0.15):
    pipeline_dir = Path(pipeline_dir)
    output_dir   = Path(output_dir)

    # Output structure
    train_img = output_dir / "images" / "train"
    val_img   = output_dir / "images" / "val"
    train_lbl = output_dir / "labels" / "train"
    val_lbl   = output_dir / "labels" / "val"
    for d in [train_img, val_img, train_lbl, val_lbl]:
        d.mkdir(parents=True, exist_ok=True)

    # Gather all (image, label) pairs from every per-video 4_dataset folder
    all_pairs = []
    dataset_dirs = sorted(pipeline_dir.glob("*/4_dataset"))
    print(f"\n{'='*55}")
    print("Merging YOLO datasets")
    print(f"{'='*55}")
    print(f"Found {len(dataset_dirs)} dataset directories:\n")

    for ds in dataset_dirs:
        video_name = ds.parent.name
        pairs_from_video = []

        for split in ("train", "val"):
            img_dir = ds / "images" / split
            lbl_dir = ds / "labels" / split
            if not img_dir.exists():
                continue
            for img_path in sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png")):
                lbl_path = lbl_dir / img_path.with_suffix(".txt").name
                if lbl_path.exists():
                    pairs_from_video.append((img_path, lbl_path))

        print(f"  {video_name:20s} -> {len(pairs_from_video):3d} labelled images")
        all_pairs.extend(pairs_from_video)

    print(f"\n  Total collected: {len(all_pairs)} image-label pairs")

    # Shuffle and split
    random.seed(42)
    random.shuffle(all_pairs)
    n_val   = max(1, int(len(all_pairs) * val_split))
    val_pairs   = all_pairs[:n_val]
    train_pairs = all_pairs[n_val:]

    print(f"  Train: {len(train_pairs)}  |  Val: {len(val_pairs)}")

    # Copy files — prefix with source video name to avoid collisions
    def copy_pairs(pairs, img_dst, lbl_dst):
        for img, lbl in pairs:
            # Use the video folder as a prefix to ensure unique filenames
            video_prefix = img.parents[2].name   # e.g. "test1"
            stem = f"{video_prefix}_{img.stem}"
            shutil.copy2(img, img_dst / f"{stem}{img.suffix}")
            shutil.copy2(lbl, lbl_dst / f"{stem}.txt")

    copy_pairs(train_pairs, train_img, train_lbl)
    copy_pairs(val_pairs,   val_img,   val_lbl)

    # Write data.yaml
    yaml_path = output_dir / "data.yaml"
    yaml_content = f"""# KitchEye — Combined dataset (all videos merged)
path: {output_dir.resolve()}
train: images/train
val:   images/val

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""
    yaml_path.write_text(yaml_content)

    print(f"\n  data.yaml written: {yaml_path}")
    print(f"\n{'='*55}")
    print("[DONE] Merge complete!")
    print(f"{'='*55}")
    print(f"  Combined dataset: {output_dir}")
    print(f"\n  To train:")
    print(f"  yolo train data={yaml_path} model=yolov8m.pt epochs=100 imgsz=640 batch=16 device=0")
    return str(yaml_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge per-video YOLO datasets")
    parser.add_argument("--pipeline-dir", default="./pipeline_output",
                        help="Root pipeline_output directory")
    parser.add_argument("--output", default="./pipeline_output/combined_dataset",
                        help="Output directory for merged dataset")
    parser.add_argument("--val-split", type=float, default=0.15,
                        help="Fraction of images to use for validation (default: 0.15)")
    args = parser.parse_args()

    merge_datasets(args.pipeline_dir, args.output, args.val_split)
