#!/usr/bin/env python3
"""
Interactive mask selector with click-based selection using OpenCV.

Usage:
    python mask_click_selector.py <frame_name>

- Displays a combined overlay of all masks on the original image.
- Click on any region to toggle selection of the corresponding mask.
- Selected masks are highlighted in red; unselected masks are semi-transparent.
- Press 's' to save the merged mask to training_masks/<frame_name>.png.
  • If no masks are selected, a blank mask (all zeros) will be saved.
- Press 'c' to clear all selections.
- Press 'q' to quit without saving (or after saving).
"""
import os
import sys
import glob
import cv2
import numpy as np

# Configuration directories
dirs = {
    "sam_masks": "/home/jaden/projects/sew/segment/tao/data/sam_masks",
    "orig_images": "/home/jaden/projects/sew/segment/tao/data/resized_images",
    "output": "/home/jaden/projects/sew/segment/tao/data/training_masks"
}
# Ensure output directory exists
os.makedirs(dirs['output'], exist_ok=True)

# Load original image for a frame
def load_original(frame):
    base = os.path.splitext(frame)[0]
    for ext in ['.png', '.jpg', '.jpeg']:
        path = os.path.join(dirs['orig_images'], base + ext)
        if os.path.isfile(path):
            img = cv2.imread(path)
            if img is not None:
                return img
    return None

# Get mask file paths for a frame
def list_masks(frame):
    folder = os.path.join(dirs['sam_masks'], frame)
    return sorted(glob.glob(os.path.join(folder, '*_mask*.png')))

# Load mask images into boolean arrays
def load_masks(mask_paths):
    masks = []
    for p in mask_paths:
        m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if m is not None:
            masks.append(m > 0)
    return masks

# Generate pastel colors for unselected masks
def generate_colors(n):
    rng = np.random.RandomState(0)
    colors = []
    for _ in range(n):
        c = rng.randint(100, 200, size=3).tolist()
        colors.append(tuple(int(x) for x in c))
    return colors

# Create overlay image with selected masks in red

def make_overlay(orig, masks, colors, selected):
    overlay = orig.copy().astype(np.float32)
    for idx, mask in enumerate(masks):
        mask_bool = mask
        if idx in selected:
            color = np.array((0, 0, 255), dtype=np.float32)  # red
            alpha = 0.6
        else:
            color = np.array(colors[idx], dtype=np.float32)
            alpha = 0.3
        overlay[mask_bool] = overlay[mask_bool] * (1 - alpha) + color * alpha
    return overlay.astype(np.uint8)

# Mouse click callback: toggle selection
def on_mouse(event, x, y, flags, param):
    global selected
    masks, colors, orig = param
    if event == cv2.EVENT_LBUTTONDOWN:
        for idx, mask in enumerate(masks):
            if mask[y, x]:
                if idx in selected:
                    selected.remove(idx)
                    print(f"Deselected mask {idx}")
                else:
                    selected.add(idx)
                    print(f"Selected mask {idx}")
                break

# Main loop
def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    frame = sys.argv[1]
    orig = load_original(frame)
    if orig is None:
        print(f"Error: original for frame '{frame}' not found.")
        sys.exit(1)

    mask_paths = list_masks(frame)
    if not mask_paths:
        print(f"Error: no masks found for frame '{frame}'.")
        sys.exit(1)

    masks = load_masks(mask_paths)
    colors = generate_colors(len(masks))
    global selected
    selected = set()

    window_name = f"Select masks: {frame}"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse, param=(masks, colors, orig))

    while True:
        img_overlay = make_overlay(orig, masks, colors, selected)
        status_text = f"Selected: {sorted(selected)} | (s=save blank/all q=quit c=clear)"
        display = cv2.putText(
            img_overlay.copy(), status_text, (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        cv2.imshow(window_name, display)
        key = cv2.waitKey(100) & 0xFF
        if key == ord('q'):
            print("Quitting.")
            break
        elif key == ord('c'):
            selected.clear()
            print("Cleared all selections.")
        elif key == ord('s'):
            # Save mask: if none selected, save blank mask
            if not selected:
                print("No masks selected; saving blank mask.")
                blank = np.zeros(orig.shape[:2], dtype=np.uint8)
                out_path = os.path.join(dirs['output'], f"{frame}.png")
                cv2.imwrite(out_path, blank)
                print(f"Saved blank mask to {out_path}")
            else:
                merged = (np.any([masks[i] for i in selected], axis=0) * 255).astype(np.uint8)
                out_path = os.path.join(dirs['output'], f"{frame}.png")
                cv2.imwrite(out_path, merged)
                print(f"Saved merged mask to {out_path}")
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
