#!/usr/bin/env python3
"""Download and load the exact ONNX models used by the web service."""
from __future__ import annotations

import json
import os
import time

import numpy as np


def main() -> None:
    from fast_alpr import ALPR

    started = time.perf_counter()
    model = ALPR(
        detector_model=os.getenv(
            "PLATES_DETECTOR_MODEL", "yolo-v9-t-384-license-plate-end2end"
        ),
        ocr_model=os.getenv("PLATES_OCR_MODEL", "cct-xs-v2-global-model"),
        detector_conf_thresh=0.35,
        ocr_device="cpu",
    )
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
