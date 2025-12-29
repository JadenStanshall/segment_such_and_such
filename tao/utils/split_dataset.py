#!/usr/bin/env python3
"""
Split a dataset of images and masks into train/val/test folders.

Usage:
    python split_dataset.py \
      --images_dir data/resized_images \
      --masks_dir data/training_masks \
      --output_dir data/training_data \
      --train_frac 0.8 \
      --val_frac 0.1 \
      [--test_frac 0.1] \
      [--seed 42]

This will create:
    data/training_data/
      train/images, train/masks
      val/images,   val/masks
      test/images,  test/masks

Images and masks must share the same base filename (e.g. frame_0001.png).
"""
import os
import argparse
import random
import shutil
from glob import glob

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--images_dir', required=True, help='Folder of source images')
    p.add_argument('--masks_dir', required=True, help='Folder of source masks')
    p.add_argument('--output_dir', required=True, help='Root output folder for train/val/test splits')
    p.add_argument('--train_frac', type=float, required=True, help='Fraction for training set (e.g. 0.8)')
    p.add_argument('--val_frac', type=float, required=True, help='Fraction for validation set (e.g. 0.1)')
    p.add_argument('--test_frac', type=float, default=None, help='Fraction for test set; if omitted, computed as 1-train-val')
    p.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    return p.parse_args()


def make_dirs(base):
    for split in ['train', 'val', 'test']:
        for sub in ['images', 'masks']:
            path = os.path.join(base, split, sub)
            os.makedirs(path, exist_ok=True)


def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    # Compute test fraction
    if args.test_frac is None:
        test_frac = 1.0 - args.train_frac - args.val_frac
    else:
        test_frac = args.test_frac
    if test_frac < 0:
        raise ValueError('train_frac + val_frac must be <= 1.0')

    # Gather images and matching masks
    img_paths = sorted(glob(os.path.join(args.images_dir, '*.*')))
    mask_paths = glob(os.path.join(args.masks_dir, '*.*'))
    mask_map = {os.path.splitext(os.path.basename(p))[0]: p for p in mask_paths}

    pairs = []
    for img in img_paths:
        base = os.path.splitext(os.path.basename(img))[0]
        mask = mask_map.get(base)
        if mask:
            pairs.append((img, mask))
        else:
            print(f'Warning: no mask found for image {base}')

    if not pairs:
        raise RuntimeError('No image-mask pairs found!')

    # Shuffle and split
    random.shuffle(pairs)
    n = len(pairs)
    n_train = int(args.train_frac * n)
    n_val = int(args.val_frac * n)
    n_test = n - n_train - n_val

    splits = {
        'train': pairs[:n_train],
        'val':   pairs[n_train:n_train+n_val],
        'test':  pairs[n_train+n_val:]
    }

    # Create output dirs
    make_dirs(args.output_dir)

    # Copy files
    for split, items in splits.items():
        for img, msk in items:
            dst_img = os.path.join(args.output_dir, split, 'images', os.path.basename(img))
            dst_msk = os.path.join(args.output_dir, split, 'masks',  os.path.basename(msk))
            shutil.copy2(img, dst_img)
            shutil.copy2(msk, dst_msk)
    print(f"Dataset split: {n_train} train, {n_val} val, {n_test} test pairs.")

if __name__ == '__main__':
    main()
