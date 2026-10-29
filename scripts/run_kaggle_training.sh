#!/usr/bin/env bash
# Explicit opt-in training: never replaces production or Open Images runs.
set -euo pipefail
APP_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
STATE_ROOT=${PLATES_KAGGLE_ROOT:-/storage/plates-kaggle}
PYTHON=${PLATES_TRAINING_PYTHON:-/var/lib/plates-training/venv/bin/python}
DEVICE=${PLATES_DEVICE:-cpu}
BATCH=${PLATES_BATCH:-1}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export YOLO_CONFIG_DIR="$STATE_ROOT/yolo-config"
export XDG_CACHE_HOME="$STATE_ROOT/cache" TMPDIR="$STATE_ROOT/tmp"
export YOLO_AUTOINSTALL=false
mkdir -p "$TMPDIR" "$YOLO_CONFIG_DIR" "$STATE_ROOT/runs"
cd "$STATE_ROOT"
DATA="$STATE_ROOT/dataset/dataset.yaml"
test -f "$DATA"
if [ ! -f "$STATE_ROOT/runs/smoke/weights/best.pt" ]; then
  "$PYTHON" "$APP_ROOT/scripts/train_detector.py" --data "$DATA" \
    --project "$STATE_ROOT/runs" --name smoke --epochs 1 --fraction 0.03 --device "$DEVICE" --batch 1
fi
if [ ! -f "$STATE_ROOT/runs/kaggle-detector-status.json" ]; then
  "$PYTHON" "$APP_ROOT/scripts/train_detector.py" --data "$DATA" \
    --project "$STATE_ROOT/runs" --name kaggle-detector --epochs 40 --resume --device "$DEVICE" --batch "$BATCH"
fi
"$PYTHON" "$APP_ROOT/scripts/evaluate_detector.py" --data "$DATA" \
  --weights "$STATE_ROOT/runs/kaggle-detector/weights/best.pt" \
  --output "$STATE_ROOT/evaluation.json"
