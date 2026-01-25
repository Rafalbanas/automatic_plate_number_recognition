from __future__ import annotations

import os
from pathlib import Path

import torch
from ultralytics import YOLO  # pyright: ignore[reportPrivateImportUsage]


def pick_device() -> str:
    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def must_exist(path: str) -> None:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing path: {p.resolve()}")


def main():
    device = pick_device()
    print(f"[INFO] Using device: {device}")

    data_yaml = "data/data.yaml"
    must_exist(data_yaml)

    # (opcjonalnie) sprawdź czy są pliki train/val
    # Uwaga: te ścieżki wynikają z data.yaml - więc to tylko szybki sanity check.
    if Path("data/splits/images/train").exists():
        n_train = len(list(Path("data/splits/images/train").glob("**/*.jpg")))
        print(f"[INFO] train images: {n_train}")
    if Path("data/splits/images/val").exists():
        n_val = len(list(Path("data/splits/images/val").glob("**/*.jpg")))
        print(f"[INFO] val images: {n_val}")

    model = YOLO("yolov8n.pt")

    results = model.train(
        data=data_yaml,
        epochs=30,
        imgsz=640,
        batch=16,
        device=device,
        project="runs/plate_det",
        name="yolov8n",
        exist_ok=True,
        patience=15,
        workers=0,   # na Macu bywa stabilniej
        cache=True,
    )

    print("[INFO] Train finished.")
    print(results)


if __name__ == "__main__":
    main()