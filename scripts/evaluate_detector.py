#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from ultralytics import YOLO

    config = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    dataset_root = Path(config["path"])
    if not dataset_root.is_absolute():
        dataset_root = args.data.parent / dataset_root
    test_dir = dataset_root / config["test"]
    test_images = sum(
        path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        for path in test_dir.iterdir()
        if path.is_file()
    )

    model = YOLO(str(args.weights))
    started = time.perf_counter()
    result = model.val(
        data=str(args.data),
        split="test",
        imgsz=416,
        batch=1,
        workers=0,
        device="cpu",
        plots=False,
        verbose=True,
        project=str(args.output.parent / "runs"),
        name="test-evaluation",
        exist_ok=True,
    )
    elapsed = time.perf_counter() - started
    metrics = {
        "test_images": test_images,
        "map50": float(result.box.map50),
        "map50_95": float(result.box.map),
        "precision": float(result.box.mp),
        "recall": float(result.box.mr),
        "elapsed_seconds": round(elapsed, 3),
        "mean_seconds_per_image": round(elapsed / test_images, 5) if test_images else None,
        "weights": str(args.weights),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
