set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ts=$(date +"%y%m%d_%H%M%S")

# Use paths relative to the project root so TAO can write inside the mounted workspace.
RESULTS_DIR="tao/output/results/${ts}"
INFER_DIR="tao/output/inference/${ts}"

mkdir -p "$ROOT_DIR/$RESULTS_DIR" "$ROOT_DIR/$INFER_DIR"
cd "$ROOT_DIR"

# Ensure the TAO container mounts and runs from the project root so relative paths resolve.
export TAO_DOCKER_OPTIONS="--volume ${ROOT_DIR}:/workspace/tao --workdir /workspace/tao"
SPEC_PATH="/workspace/tao/utils/specs/segment_spec.txt"

tao model unet train \
    -r "$RESULTS_DIR" \
    -e "$SPEC_PATH" \
    -k tlt_encode \
    -n freespace 

tao model unet inference \
    -e "$SPEC_PATH" \
    -m "${RESULTS_DIR}/weights/freespace.tlt" \
    -i /workspace/tao/data/dataset/test/images \
    -o "$INFER_DIR"
