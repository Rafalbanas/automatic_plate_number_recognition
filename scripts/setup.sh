#!/usr/bin/env bash
set -euo pipefail

PYTHON="python3"

# 1) venv
if [ ! -d ".venv" ]; then
  "$PYTHON" -m venv .venv
fi

# 2) pip + requirements
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt

echo "✅ Gotowe. Aktywuj venv: source .venv/bin/activate"