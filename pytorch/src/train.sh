#!/usr/bin/env bash

DATA_ROOT="/home/jaden/projects/sew/segment/pytorch/data"
RESULTS_DIR="/home/jaden/projects/sew/segment/pytorch/src/results"

EPOCHS=10
BATCH_SIZE=2
LR=1e-5

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

mkdir -p "$RESULTS_DIR"

# run training
cd /home/jaden/projects/sew/segment/pytorch/src
python3 train.py \
  --data-root "$DATA_ROOT" \
  --results-dir "$RESULTS_DIR" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR"

# run eval
cd /home/jaden/projects/sew/segment/pytorch/src
python3 eval.py \
  --data-root "$DATA_ROOT" \
  --checkpoint /home/jaden/projects/sew/segment/pytorch/src/results/checkpoints/unet_epoch10.pth \
  --batch-size 1 \
  --thresh 0.5
