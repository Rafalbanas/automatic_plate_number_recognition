#!/usr/bin/env python3
"""Download and load the exact ONNX models used by the web service."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    from webapp.inference import create_alpr

    started = time.perf_counter()
    model = create_alpr()
    result = model.predict(np.zeros((384, 384, 3), dtype=np.uint8))
    print(
        json.dumps(
            {
                "status": "ready",
                "warmup_detections": len(result),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
