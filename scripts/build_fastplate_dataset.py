from __future__ import annotations
import os
import csv
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np


def crop_safe(img: np.ndarray, xyxy: tuple[float, float, float, float]) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return img[0:1, 0:1].copy()
    return img[y1:y2, x1:x2].copy()


def pad_xyxy(xyxy: tuple[float, float, float, float], pad: float = 0.15) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = xyxy
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    dx = bw * pad
    dy = bh * pad
    return (x1 - dx, y1 - dy, x2 + dx, y2 + dy)


def find_image_path(images_root: Path, name: str) -> Path | None:
    # najpierw bezpośrednio
    p = images_root / name
    if p.exists():
        return p
    # fallback: szukanie rekurencyjne (wolniejsze, ale pewne)
    hits = list(images_root.rglob(name))
    return hits[0] if hits else None


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", required=True, help="Ścieżka do annotations.xml (CVAT)")
    ap.add_argument("--images-root", required=True, help="Folder z obrazami źródłowymi")
    ap.add_argument("--out", default="data/fastplate_pl", help="Wyjściowy folder datasetu")
    ap.add_argument("--pad", type=float, default=0.15, help="Padding bbox (0.10-0.20 zwykle najlepsze)")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    args = ap.parse_args()

    random.seed(args.seed)

    ann_path = Path(args.annotations)
    images_root = Path(args.images_root)
    out_dir = Path(args.out)
    crops_dir = out_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(ann_path)
    root = tree.getroot()

    rows: list[tuple[str, str]] = []  # (rel_image_path, text)

    idx = 0
    for im in root.findall(".//image"):
        name = im.attrib.get("name", "")
        img_path = find_image_path(images_root, name)
        if img_path is None:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        for box in im.findall("box"):
            if box.attrib.get("label") != "plate":
                continue

            xtl = float(box.attrib["xtl"])
            ytl = float(box.attrib["ytl"])
            xbr = float(box.attrib["xbr"])
            ybr = float(box.attrib["ybr"])
            xyxy = pad_xyxy((xtl, ytl, xbr, ybr), pad=args.pad)

            # tekst tablicy
            attr = box.find("attribute")
            if attr is None or (attr.text is None):
                continue
            text = attr.text.strip().upper()
            if not text:
                continue

            crop = crop_safe(img, xyxy)
            crop = np.ascontiguousarray(crop, dtype=np.uint8)

            out_name = f"plate_{idx:06d}.jpg"
            out_path = crops_dir / out_name
            cv2.imwrite(str(out_path), crop)

            rel = str(Path("crops") / out_name)
            rows.append((rel, text))
            idx += 1

    if not rows:
        raise SystemExit("Nie udało się wygenerować żadnych cropów. Sprawdź images-root i annotations.")

    random.shuffle(rows)
    n_val = int(len(rows) * args.val_ratio)
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]

    def write_csv(p: Path, rr: list[tuple[str, str]]):
        # fast-plate-ocr wymaga dokładnie dwóch kolumn: image_path i plate_text
        with p.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image_path", "plate_text"])
            for img_rel, txt in rr:
                w.writerow([img_rel, txt])

    write_csv(out_dir / "train_annotations.csv", train_rows)
    write_csv(out_dir / "valid_annotations.csv", val_rows)

    print(f"OK. crops: {len(rows)}  train: {len(train_rows)}  val: {len(val_rows)}")
    print(f"Dataset: {out_dir}")


if __name__ == "__main__":
    main()