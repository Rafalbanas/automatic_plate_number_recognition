#!/usr/bin/env bash
set -euo pipefail

APP_ROOT=${PLATES_APP_ROOT:-/opt/plates-web}
STATE_ROOT=${PLATES_TRAINING_ROOT:-/var/lib/plates-training}
PYTHON="$STATE_ROOT/venv/bin/python"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PIP_NO_CACHE_DIR=1

python3 -m venv "$STATE_ROOT/venv"
"$PYTHON" -m pip install --disable-pip-version-check \
  --index-url https://download.pytorch.org/whl/cpu \
  torch==2.14.1+cpu torchvision==0.29.1+cpu
"$PYTHON" -m pip install --disable-pip-version-check \
  -r "$APP_ROOT/requirements-train.txt"
# Ultralytics declares the GUI OpenCV wheel, but this headless VPS has no
# libGL. Replace it after dependency resolution without changing the host.
"$PYTHON" -m pip uninstall -y opencv-python
"$PYTHON" -m pip install --disable-pip-version-check --no-deps --force-reinstall \
  opencv-python-headless==5.0.0.93
"$PYTHON" "$APP_ROOT/scripts/prepare_openimages.py" \
  --root "$STATE_ROOT/dataset" --train-limit 600 --validation-limit 240
