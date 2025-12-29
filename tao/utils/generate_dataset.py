#!/usr/bin/env python3
"""
Create train/val/test splits from paired images and masks.

Defaults:
  images: data/resized_images
  masks:  data/masks
  output: data/dataset
  split:  0.8/0.1/0.1

Example:
  python utils/generate_dataset.py \
    --images_dir data/resized_images \
    --masks_dir data/masks \
    --output_root data/dataset \
    --train_frac 0.8 --val_frac 0.1 --test_frac 0.1 \
    --seed 42
"""
import argparse
import os
import random
import shutil
from glob import glob


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images_dir", default="data/resized_images", help="Folder containing source images")
    p.add_argument("--masks_dir", default="data/training_masks", help="Folder containing source masks")
    p.add_argument("--output_root", default="data/dataset", help="Root folder for train/val/test splits")
    p.add_argument("--train_frac", type=float, default=0.8, help="Fraction for training split")
    p.add_argument("--val_frac", type=float, default=0.1, help="Fraction for validation split")
    p.add_argument("--test_frac", type=float, default=0.1, help="Fraction for test split")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return p.parse_args()


def resolve_path(path, root):
    """Resolve path relative to repo root if not absolute."""
    return path if os.path.isabs(path) else os.path.join(root, path)


def make_dirs(root):
    for split in ["train", "val", "test"]:
        for sub in ["images", "masks"]:
            os.makedirs(os.path.join(root, split, sub), exist_ok=True)


def collect_pairs(images_dir, masks_dir):
    img_paths = sorted(glob(os.path.join(images_dir, "*.*")))
    mask_paths = glob(os.path.join(masks_dir, "*.*"))
    mask_map = {os.path.splitext(os.path.basename(p))[0]: p for p in mask_paths}

    pairs = []
    for img in img_paths:
        base = os.path.splitext(os.path.basename(img))[0]
        mask = mask_map.get(base)
        if mask:
            pairs.append((img, mask))
        else:
            print(f"Warning: no mask found for image '{base}'")
    if not pairs:
        raise RuntimeError(f"No image-mask pairs found. Checked images_dir='{images_dir}', masks_dir='{masks_dir}'.")
    return pairs


def main():
    args = parse_args()
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    images_dir = resolve_path(args.images_dir, repo_root)
    masks_dir = resolve_path(args.masks_dir, repo_root)
    output_root = resolve_path(args.output_root, repo_root)

    if args.train_frac + args.val_frac + args.test_frac > 1.000001:
        raise ValueError("train_frac + val_frac + test_frac must be <= 1.0")

    random.seed(args.seed)
    pairs = collect_pairs(images_dir, masks_dir)
    random.shuffle(pairs)

    n = len(pairs)
    n_train = int(args.train_frac * n)
    n_val = int(args.val_frac * n)
    n_test = n - n_train - n_val

    splits = {
        "train": pairs[:n_train],
        "val": pairs[n_train:n_train + n_val],
        "test": pairs[n_train + n_val:],
    }

    make_dirs(output_root)

    for split, items in splits.items():
        for img, msk in items:
            dst_img = os.path.join(output_root, split, "images", os.path.basename(img))
            dst_msk = os.path.join(output_root, split, "masks", os.path.basename(msk))
            shutil.copy2(img, dst_img)
            shutil.copy2(msk, dst_msk)

    print(f"Done. Saved {n_train} train, {n_val} val, {n_test} test pairs to '{output_root}'.")


if __name__ == "__main__":
    main()
