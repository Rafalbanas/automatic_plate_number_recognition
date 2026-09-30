#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Resource-conservative YOLO plate training")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")

    from ultralytics import YOLO

    last = args.project / args.name / "weights" / "last.pt"
    started = time.time()
    if args.resume and last.is_file():
        print(f"[train] resuming {last}", flush=True)
        model = YOLO(str(last))
        model.train(resume=True)
    else:
        print("[train] starting from official yolo11n.pt pretrained weights", flush=True)
        # The systemd service has a read-only application directory. Keep the
        # downloaded upstream weights in the writable training state instead.
        pretrained = args.project.parent / "yolo11n.pt"
        model = YOLO(str(pretrained))
        model.train(
            data=str(args.data),
            epochs=args.epochs,
            imgsz=416,
            batch=1,
            workers=0,
            cache=False,
            device="cpu",
            amp=False,
            project=str(args.project),
            name=args.name,
            exist_ok=True,
            patience=12,
            save=True,
            save_period=1,
            fraction=args.fraction,
            deterministic=True,
            seed=2026,
            single_cls=True,
            plots=False,
            verbose=True,
        )
    summary = {"name": args.name, "elapsed_seconds": round(time.time() - started, 3), "last_checkpoint": str(last)}
    (args.project / f"{args.name}-status.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
