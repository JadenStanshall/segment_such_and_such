#!/usr/bin/env bash
set -euo pipefail

# create venv
VENV_DIR="/home/jaden/projects/sew/segment/tao/.venv"
if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtualenv in $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"


# install deps
pip install --upgrade pip setuptools wheel
pip install -r /home/jaden/projects/sew/segment/tao/requirements.txt


distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -sL https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker


# get sam weights
WEIGHTS_DIR="/home/jaden/projects/sew/segment/tao/utils/weights"
WEIGHTS_FILE="$WEIGHTS_DIR/sam_vit_h_4b8939.pth"
if [ ! -f "$WEIGHTS_FILE" ]; then
  mkdir -p "$WEIGHTS_DIR"
  echo "Downloading SAM ViT-H weights into $WEIGHTS_FILE"
  wget -O "$WEIGHTS_FILE" \
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
fi

echo
echo "to launch environment, run:  source $VENV_DIR/bin/activate"
