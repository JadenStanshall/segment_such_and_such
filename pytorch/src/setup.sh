#!/usr/bin/env bash
set -euo pipefail

# create venv
VENV_DIR="/home/jaden/projects/sew/segment/pytorch/.venv"
if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtualenv in $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# install deps
pip install --upgrade pip setuptools wheel
pip install -r /home/jaden/projects/sew/segment/pytorch/requirements.txt

echo
echo "to launch environment, run:  source $VENV_DIR/bin/activate"
