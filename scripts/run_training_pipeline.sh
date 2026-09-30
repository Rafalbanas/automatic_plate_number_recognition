#!/usr/bin/env bash
set -euo pipefail

APP_ROOT=${PLATES_APP_ROOT:-/opt/plates-web}
STATE_ROOT=${PLATES_TRAINING_ROOT:-/var/lib/plates-training}
PYTHON="$STATE_ROOT/venv/bin/python"
DATA="$STATE_ROOT/dataset/dataset.yaml"
RUNS="$STATE_ROOT/runs"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export YOLO_CONFIG_DIR="$STATE_ROOT/yolo-config"

if [ ! -f "$DATA" ]; then
  echo "Missing prepared dataset: $DATA" >&2
  exit 2
fi

if [ ! -f "$RUNS/smoke/weights/last.pt" ]; then
  "$PYTHON" "$APP_ROOT/scripts/train_detector.py" --data "$DATA" --project "$RUNS" --name smoke --epochs 1 --fraction 0.03
fi

"$PYTHON" "$APP_ROOT/scripts/train_detector.py" --data "$DATA" --project "$RUNS" --name plate-detector --epochs 40 --resume

"$PYTHON" "$APP_ROOT/scripts/evaluate_detector.py" \
  --data "$DATA" \
  --weights "$RUNS/plate-detector/weights/best.pt" \
  --output "$STATE_ROOT/evaluation.json"
