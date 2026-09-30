#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    started = time.perf_counter()
    result = model.val(data=str(args.data), split="test", imgsz=416, batch=1, workers=0, device="cpu", plots=False, verbose=True)
    elapsed = time.perf_counter() - started
    metrics = {
        "test_images": int(result.seen),
        "map50": float(result.box.map50),
        "map50_95": float(result.box.map),
        "precision": float(result.box.mp),
        "recall": float(result.box.mr),
        "elapsed_seconds": round(elapsed, 3),
        "mean_seconds_per_image": round(elapsed / result.seen, 5) if result.seen else None,
        "weights": str(args.weights),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
