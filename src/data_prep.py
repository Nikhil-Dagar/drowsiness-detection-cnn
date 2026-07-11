"""
data_prep.py
One-time helper: splits a flat dataset of Open/Closed eye images into
train/ and val/ folders expected by train_model.py.

If your downloaded dataset already looks like:
    raw_dataset/
        Open/*.jpg
        Closed/*.jpg

Run:
    python src/data_prep.py --src raw_dataset --dst dataset --val_split 0.2
"""

import argparse
import os
import random
import shutil


def split_class(src_class_dir, dst_train_dir, dst_val_dir, val_split, seed):
    files = [f for f in os.listdir(src_class_dir)
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    random.Random(seed).shuffle(files)

    n_val = int(len(files) * val_split)
    val_files = files[:n_val]
    train_files = files[n_val:]

    os.makedirs(dst_train_dir, exist_ok=True)
    os.makedirs(dst_val_dir, exist_ok=True)

    for fname in train_files:
        shutil.copy2(os.path.join(src_class_dir, fname), os.path.join(dst_train_dir, fname))
    for fname in val_files:
        shutil.copy2(os.path.join(src_class_dir, fname), os.path.join(dst_val_dir, fname))

    return len(train_files), len(val_files)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Raw dataset dir with Open/ and Closed/ subfolders")
    parser.add_argument("--dst", default="dataset", help="Output dataset root")
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for cls in ("Open", "Closed"):
        src_class_dir = os.path.join(args.src, cls)
        if not os.path.isdir(src_class_dir):
            raise FileNotFoundError(f"Expected folder not found: {src_class_dir}")

        n_train, n_val = split_class(
            src_class_dir,
            os.path.join(args.dst, "train", cls),
            os.path.join(args.dst, "val", cls),
            args.val_split,
            args.seed,
        )
        print(f"{cls}: {n_train} train, {n_val} val")

    print(f"\nDone. Dataset ready at: {args.dst}/train and {args.dst}/val")


if __name__ == "__main__":
    main()
