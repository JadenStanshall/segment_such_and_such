#!/usr/bin/env bash
set -euo pipefail

# Activate project virtualenv if present
if [ -f ".venv/bin/activate" ]; then
  source ".venv/bin/activate"
fi

# === Configuration (edit these) ===
VIDEO_PATH="/home/jaden/projects/sew/segment/tao/data/input_video/test_video.mp4"
INTERVAL_SEC=1.0
RAW_DIR="/home/jaden/projects/sew/segment/tao/data/raw_images"
RESIZED_DIR="/home/jaden/projects/sew/segment/tao/data/resized_images"
TARGET_WIDTH=640       # set to an integer for width (px), or leave empty
TARGET_HEIGHT=""      # set to an integer for height (px), or leave empty
# ================================

# 1) Extract frames
echo "Extracting frames from '$VIDEO_PATH' every $INTERVAL_SEC seconds into '$RAW_DIR'..."
python /home/jaden/projects/sew/segment/tao/utils/video_to_frames.py "$VIDEO_PATH" --interval "$INTERVAL_SEC" --output_dir "$RAW_DIR"

# 2) Resize images
echo "Resizing images from '$RAW_DIR' into '$RESIZED_DIR'..."

resize_cmd=(python /home/jaden/projects/sew/segment/tao/utils/resize_images.py --input_dir "$RAW_DIR" --output_dir "$RESIZED_DIR")

if [ -n "$TARGET_WIDTH" ]; then
  resize_cmd+=(--width "$TARGET_WIDTH")
fi
if [ -n "$TARGET_HEIGHT" ]; then
  resize_cmd+=(--height "$TARGET_HEIGHT")
fi

"${resize_cmd[@]}"

echo "Done! Frames in '$RAW_DIR', resized images in '$RESIZED_DIR'."
