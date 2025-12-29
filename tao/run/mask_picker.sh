#!/usr/bin/env bash
set -euo pipefail

# Activate project virtualenv if present
if [ -f ".venv/bin/activate" ]; then
  source ".venv/bin/activate"
fi

PROJ_DIR="/home/jaden/projects/sew/segment/tao"

# Ensure the sam_masks directory exists
if [ ! -d "$PROJ_DIR/data/sam_masks" ]; then
  echo "Error: 'sam_masks' directory not found."
  exit 1
fi

# Loop over each frame folder in sam_masks
for frame in $(ls "$PROJ_DIR/data/sam_masks"); do
  echo "======================================"
  echo "Processing frame: $frame"
  echo "(Press 's' to save merged mask, 'q' to skip and continue)"
  # Invoke the interactive selector for this frame
  python "$PROJ_DIR/utils/mask_picker.py" "$frame"
  echo "Finished $frame"
  echo
# After each selection, script will continue to next frame
# User can quit individual frame without saving by pressing 'q'
done

echo "All frames have been processed."
