#!/usr/bin/env bash
set -euo pipefail

# Activate project virtualenv if present
if [ -f ".venv/bin/activate" ]; then
  source ".venv/bin/activate"
fi

# === Configuration (edit as needed) ===
IMAGES_DIR="/home/jaden/projects/sew/segment/data/resized_images"   # Folder of source images
MASKS_DIR="/home/jaden/projects/sew/segment/data/training_masks"   # Folder of source masks
OUTPUT_DIR="/home/jaden/projects/sew/segment/data/dataset"    # Root output for train/val/test splits
TRAIN_FRAC=0.8                      # Fraction for training set
VAL_FRAC=0.1                        # Fraction for validation set
SEED=42                             # Random seed for reproducibility
# ================================

# Run the Python split script
echo "Splitting dataset into train/val/test..."
python /home/jaden/projects/sew/segment/utils/split_dataset.py \
  --images_dir "$IMAGES_DIR" \
  --masks_dir "$MASKS_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --train_frac "$TRAIN_FRAC" \
  --val_frac "$VAL_FRAC" \
  --seed "$SEED"

echo "Dataset split complete. Output in '$OUTPUT_DIR'."
