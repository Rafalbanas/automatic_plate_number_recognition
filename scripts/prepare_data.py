#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple


@dataclass
class Ann:
    filename: str
    width: int
    height: int
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    plate_text: str


def parse_cvat_xml(xml_path: Path, label_name: str = "plate", attr_name: str = "plate number") -> List[Ann]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    out: List[Ann] = []

    for img in root.findall("image"):
        fname = img.get("name")
        w_str = img.get("width")
        h_str = img.get("height")

        if not fname or not w_str or not h_str:
            continue

        w = int(float(w_str))
        h = int(float(h_str))

        # w CVAT może być 0/1/many boxów na obraz
        for box in img.findall("box"):
            if box.get("label") != label_name:
                continue

            xtl_str = box.get("xtl")
            ytl_str = box.get("ytl")
            xbr_str = box.get("xbr")
            ybr_str = box.get("ybr")
            if not all([xtl_str, ytl_str, xbr_str, ybr_str]):
                continue

            def req_attr(el: ET.Element, key: str) -> str:
                v = el.get(key)
                if v is None:
                    raise ValueError(f"Brak atrybutu '{key}' w tagu <{el.tag}>")
                return v
            xtl = float(req_attr(box, "xtl"))
            ytl = float(req_attr(box, "ytl"))
            xbr = float(req_attr(box, "xbr"))
            ybr = float(req_attr(box, "ybr"))

            # atrybut z numerem tablicy
            plate_text = ""
            for attr in box.findall("attribute"):
                if attr.get("name") == attr_name:
                    plate_text = (attr.text or "").strip()
                    break

            # sanity clamp
            xmin = max(0.0, min(xtl, xbr))
            ymin = max(0.0, min(ytl, ybr))
            xmax = min(float(w), max(xtl, xbr))
            ymax = min(float(h), max(ytl, ybr))

            out.append(Ann(
                filename=fname,
                width=w,
                height=h,
                xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax,
                plate_text=plate_text
            ))

    if not out:
        raise RuntimeError("Nie znaleziono żadnych adnotacji 'plate' w XML. Sprawdź label_name/attr_name.")
    return out


def yolo_line(a: Ann, class_id: int = 0) -> str:
    # YOLO: class cx cy w h (wszystko znormalizowane 0..1)
    cx = ((a.xmin + a.xmax) / 2.0) / a.width
    cy = ((a.ymin + a.ymax) / 2.0) / a.height
    bw = (a.xmax - a.xmin) / a.width
    bh = (a.ymax - a.ymin) / a.height

    # clamp
    cx = min(1.0, max(0.0, cx))
    cy = min(1.0, max(0.0, cy))
    bw = min(1.0, max(0.0, bw))
    bh = min(1.0, max(0.0, bh))

    return f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def ensure_empty_dir(p: Path) -> None:
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


def split_files(unique_files: List[str], test_ratio: float, val_ratio: float, seed: int) -> Dict[str, List[str]]:
    rng = random.Random(seed)
    files = unique_files[:]
    rng.shuffle(files)

    n = len(files)
    n_test = int(round(n * test_ratio))
    n_test = max(0, min(n, n_test))

    test = files[:n_test]
    rest = files[n_test:]

    n_val = int(round(len(rest) * val_ratio))
    n_val = max(0, min(len(rest), n_val)) if rest else 0

    val = rest[:n_val]
    train = rest[n_val:]

    # If train is empty, but we have other files, move them to train.
    if n > 0 and not train:
        if val:
            train.append(val.pop())
        elif test:
            train.append(test.pop())

    return {"train": train, "val": val, "test": test}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=str, default="data/raw", help="Katalog z jpg + annotations.xml")
    ap.add_argument("--xml", type=str, default="", help="Ścieżka do annotations.xml (domyślnie: raw_dir/annotations.xml)")
    ap.add_argument("--out_dir", type=str, default="data/prepared", help="Wyjściowy katalog")
    ap.add_argument("--test_ratio", type=float, default=0.30, help="Test split (>= 0.30 wg wymagań)")
    ap.add_argument("--val_ratio", type=float, default=0.10, help="Val split liczony z (train+val) po odjęciu test")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    xml_path = Path(args.xml) if args.xml else (raw_dir / "poland-plates" / "annotations.xml")
    # Heurystyka do znalezienia katalogu z obrazami
    #
    # Dataset z Kaggle ma je w `photos/`. W CVAT często jest to `images/`
    # Czasem są w tym samym katalogu co XML.
    images_dir = None
    candidates = [
        raw_dir / "photos",
        raw_dir / "images",
        xml_path.parent,
        xml_path.parent / "photos",
        xml_path.parent / "images",
        raw_dir,
    ]
    for c in candidates:
        if c.exists() and any(c.glob("*.jp*g")):
            images_dir = c
            print(f"INFO: Znalazłem obrazy w '{c}'")
            break

    if images_dir is None:
        raise FileNotFoundError(
            "Nie udało się znaleźć katalogu z obrazami JPG. "
            f"Sprawdziłem {', '.join(map(str, candidates))}"
        )
    out_dir = Path(args.out_dir).resolve()

    if not raw_dir.exists():
        raise FileNotFoundError(f"Nie ma katalogu: {raw_dir}")
    if not xml_path.exists():
        raise FileNotFoundError(f"Nie ma pliku XML: {xml_path}")

    anns = parse_cvat_xml(xml_path)

    # unikalne pliki (ważne, bo może być 1 box/obraz lub więcej)
    unique_files = sorted({a.filename for a in anns})
    splits = split_files(unique_files, test_ratio=args.test_ratio, val_ratio=args.val_ratio, seed=args.seed)

    # foldery
    images_root = out_dir / "images"
    labels_root = out_dir / "labels_yolo"
    ensure_empty_dir(out_dir)
    for s in ["train", "val", "test"]:
        (images_root / s).mkdir(parents=True, exist_ok=True)
        (labels_root / s).mkdir(parents=True, exist_ok=True)

    # szybki lookup adnotacji per plik
    by_file: Dict[str, List[Ann]] = {}
    for a in anns:
        by_file.setdefault(a.filename, []).append(a)

    # kopiowanie obrazów + generacja labeli
    missing = []
    for split_name, files in splits.items():
        for fname in files:
            src_img = images_dir / fname
            if not src_img.exists():
                missing.append(fname)
                continue

            dst_img = images_root / split_name / Path(fname).name
            shutil.copy2(src_img, dst_img)

            # label YOLO (ten sam stem co obraz)
            label_path = labels_root / split_name / (Path(fname).stem + ".txt")
            lines = [yolo_line(a, class_id=0) for a in by_file.get(fname, [])]
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    if missing:
        print("UWAGA: brakuje tych obrazów z XML (nie skopiowano):")
        for m in missing[:50]:
            print(" -", m)
        if len(missing) > 50:
            print(f" ... i jeszcze {len(missing)-50}")

    # zapis GT do OCR + bbox (do ewaluacji)
    gt_csv = out_dir / "gt.csv"
    with gt_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "width", "height", "xmin", "ymin", "xmax", "ymax", "plate_text"])
        for a in anns:
            w.writerow([a.filename, a.width, a.height, a.xmin, a.ymin, a.xmax, a.ymax, a.plate_text])

    # zapis splitów
    split_csv = out_dir / "split.csv"
    with split_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "split"])
        for s, files in splits.items():
            for fname in files:
                w.writerow([fname, s])

    print("OK ✅ Przygotowano dane w:", out_dir)
    print(" - obrazy:", images_root)
    print(" - labelki YOLO:", labels_root)
    print(" - gt OCR/bbox:", gt_csv)
    print(" - splity:", split_csv)
    print("Split sizes:", {k: len(v) for k, v in splits.items()})


if __name__ == "__main__":
    main()