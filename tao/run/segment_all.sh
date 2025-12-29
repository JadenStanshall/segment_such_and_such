#!/usr/bin/env bash
set -euo pipefail

# Activate project virtualenv if present
if [ -f ".venv/bin/activate" ]; then
  source ".venv/bin/activate"
fi

# === Configuration (edit these) ===
CHECKPOINT="/home/jaden/projects/sew/segment/tao/utils/weights/sam_vit_h_4b8939.pth"
INPUT_DIR="/home/jaden/projects/sew/segment/tao/data/resized_images"
OUTPUT_DIR="/home/jaden/projects/sew/segment/tao/data/sam_masks"
# Use DEVICE=cpu to avoid CUDA OOM on small GPUs. MODEL_TYPE must match the checkpoint.
DEVICE="${DEVICE:-auto}"
MODEL_TYPE="${MODEL_TYPE:-vit_h}"
# ================================

echo "Generating SAM masks from images in '$INPUT_DIR'..."
python /home/jaden/projects/sew/segment/tao/utils/segment_all.py \
  --ckpt "$CHECKPOINT" \
  --in_dir "$INPUT_DIR" \
  --out_dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --model_type "$MODEL_TYPE"

echo "All masks saved under '$OUTPUT_DIR'"
