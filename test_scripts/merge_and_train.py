"""
Merge per-video YOLO datasets into one combined dataset, then train YOLOv8.

Usage:
    python merge_and_train.py \
        --datasets  ./pipeline_output_yolo \
        --output    ./merged_dataset \
        --model     ./models/best_03062026.pt \
        --epochs    100 \
        --device    0
"""

import argparse
import shutil
import json
import random
from pathlib import Path


def merge_datasets(pipeline_root: Path, output_dir: Path) -> tuple:
    """Merge all per-video 4_dataset folders into one. Returns (data_yaml_path, class_names)."""

    train_img = output_dir / "images" / "train"
    val_img   = output_dir / "images" / "val"
    train_lbl = output_dir / "labels" / "train"
    val_lbl   = output_dir / "labels" / "val"
    for d in [train_img, val_img, train_lbl, val_lbl]:
        d.mkdir(parents=True, exist_ok=True)

    class_names = None
    nc = 0
    total_train = 0
    total_val   = 0

    dataset_dirs = sorted([
        d for d in pipeline_root.iterdir()
        if d.is_dir() and (d / "4_dataset").exists()
    ])

    if not dataset_dirs:
        raise FileNotFoundError(f"No 4_dataset folders found under {pipeline_root}")

    print(f"Merging {len(dataset_dirs)} video dataset(s)...")

    for video_dir in dataset_dirs:
        ds = video_dir / "4_dataset"

        # Read class names from this video's data.yaml
        yaml_path = ds / "data.yaml"
        if yaml_path.exists() and class_names is None:
            for line in yaml_path.read_text().splitlines():
                if line.strip().startswith("names:"):
                    raw = line.split("names:")[1].strip()
                    class_names = [c.strip().strip("'\"") for c in raw.strip("[]").split(",")]
                elif line.strip().startswith("nc:"):
                    nc = int(line.split(":")[1].strip())

        # Copy train
        for img in (ds / "images" / "train").glob("*.jpg"):
            shutil.copy2(img, train_img / img.name)
            lbl = ds / "labels" / "train" / img.with_suffix(".txt").name
            if lbl.exists():
                shutil.copy2(lbl, train_lbl / lbl.name)
            total_train += 1

        # Copy val
        for img in (ds / "images" / "val").glob("*.jpg"):
            shutil.copy2(img, val_img / img.name)
            lbl = ds / "labels" / "val" / img.with_suffix(".txt").name
            if lbl.exists():
                shutil.copy2(lbl, val_lbl / lbl.name)
            total_val += 1

        print(f"  {video_dir.name}: merged")

    print(f"\nMerged dataset: {total_train} train  |  {total_val} val")
    print(f"Classes ({len(class_names)}): {', '.join(class_names)}")

    yaml_content = f"""# KitchEye merged dataset
path: {output_dir.resolve()}
train: images/train
val:   images/val

nc: {len(class_names)}
names: {class_names}
"""
    yaml_path = output_dir / "data.yaml"
    yaml_path.write_text(yaml_content)
    print(f"data.yaml: {yaml_path}")
    return yaml_path, class_names


def train(data_yaml: Path, base_model: str, epochs: int, device: str, output_dir: Path):
    from ultralytics import YOLO

    print(f"\nBase model : {base_model}")
    print(f"Epochs     : {epochs}")
    print(f"Device     : {device}")

    model = YOLO(base_model)
    model.train(
        data     = str(data_yaml),
        epochs   = epochs,
        imgsz    = 640,
        batch    = 8,          # safe for RTX 3050 4GB
        workers  = 0,          # avoid Windows paging file exhaustion from multiprocessing
        device   = device,
        project  = str(output_dir),
        name     = "kitcheye_v2",
        exist_ok = True,
        degrees  = 45,
        flipud   = 0.5,
        fliplr   = 0.5,
        hsv_v    = 0.4,
        hsv_s    = 0.5,
        mosaic   = 1.0,
        mixup    = 0.1,
    )

    best = output_dir / "kitcheye_v2" / "weights" / "best.pt"
    print(f"\n✓ Training complete.")
    print(f"  Best model: {best}")
    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", required=True,
        help="pipeline_output_yolo folder (contains per-video subdirs)")
    parser.add_argument("--output",   default="./merged_dataset",
        help="Where to write merged dataset + trained model")
    parser.add_argument("--model",    default="./models/best_03062026.pt",
        help="Base .pt to fine-tune (default: best_03062026.pt)")
    parser.add_argument("--epochs",   type=int, default=100)
    parser.add_argument("--device",   default="0")
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_yaml, class_names = merge_datasets(Path(args.datasets), output_dir)

    if args.skip_training:
        print("\n⏭  Skipping training.")
        print(f"  Run: yolo train data={data_yaml} model={args.model} epochs={args.epochs}")
    else:
        best = train(data_yaml, args.model, args.epochs, args.device, output_dir)
        print(f"\nCopy best model to models/ folder:")
        print(f"  copy {best} .\\models\\kitcheye_v2.pt")
