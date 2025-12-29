#!/usr/bin/env python3
import os
import glob
import cv2
import argparse

def resize_images(input_dir, output_dir, width=None, height=None):
    if not (width or height):
        raise ValueError("Specify at least --width or --height")

    os.makedirs(output_dir, exist_ok=True)
    patterns = ["*.jpg", "*.jpeg", "*.png"]
    files = []
    for p in patterns:
        files += glob.glob(os.path.join(input_dir, p))
    files.sort()

    for path in files:
        img = cv2.imread(path)
        if img is None:
            print(f"⚠️  Could not read {path}")
            continue

        h, w = img.shape[:2]
        if width and height:
            new_w, new_h = width, height
        elif width:
            new_w = width
            new_h = int(h * (width / w))
        else:
            new_h = height
            new_w = int(w * (height / h))

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        fname = os.path.basename(path)
        out_path = os.path.join(output_dir, fname)
        cv2.imwrite(out_path, resized)

    print(f"Resized {len(files)} images to '{output_dir}'")

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Resize all images in a folder"
    )
    p.add_argument(
        "--input_dir", default="images",
        help="Folder of source images"
    )
    p.add_argument(
        "--output_dir", default="resized_images",
        help="Where to save resized images"
    )
    p.add_argument(
        "--width", type=int,
        help="Target width (px); preserves aspect ratio if --height is unset"
    )
    p.add_argument(
        "--height", type=int,
        help="Target height (px); preserves aspect ratio if --width is unset"
    )
    args = p.parse_args()
    resize_images(args.input_dir, args.output_dir, args.width, args.height)
