#!/usr/bin/env python3
import os
import glob
import cv2
import numpy as np

# ─── Edit this to point at your dataset root ───────────────────────────────────
DATASET_DIR = "/home/jaden/projects/sew/segment/data/dataset"
# ────────────────────────────────────────────────────────────────────────────────

for split in ("train", "val", "test"):
    src_dir = os.path.join(DATASET_DIR, split, "masks")
    dst_dir = os.path.join(DATASET_DIR, split, "masks_01")
    if not os.path.isdir(src_dir):
        print(f"[!] Skipping '{split}': source folder not found at {src_dir}")
        continue

    os.makedirs(dst_dir, exist_ok=True)
    print(f"Processing split='{split}':")
    for src_path in sorted(glob.glob(os.path.join(src_dir, "*.png"))):
        # Read grayscale mask (0 or 255)
        mask = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"   ✗ failed to read {src_path}")
            continue

        # Convert 255 → 1, leave 0 as 0
        mask01 = (mask > 128).astype(np.uint8)

        # Write out (still as PNG, values will be 0 or 1)
        fname = os.path.basename(src_path)
        dst_path = os.path.join(dst_dir, fname)
        cv2.imwrite(dst_path, mask01)

        # Optional: print unique values to verify
        uniques = np.unique(mask01)
        print(f"   ✓ {fname}: unique values {uniques.tolist()}")

    print(f"→ Remapped masks written to {dst_dir}\n")

print("All splits processed.")
