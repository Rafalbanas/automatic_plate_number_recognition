#!/usr/bin/env python3
"""Prepare the locally cached Kaggle CVAT plate dataset into leak-aware YOLO splits.

Source: Kaggle dataset ``piotrstefaskiue/poland-vehicle-license-plate-dataset``
(CVAT ``annotations.xml`` + full photos), first downloaded by kagglehub into a
local cache and passed here as ``--raw-dir``. No OCR model is used: the real
``plate number`` texts are retained verbatim in ``gt.csv`` for future
evaluation only. Every annotation is validated; broken or missing data raises
``PreparationError`` and is never silently skipped.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from prepare_openimages import group_similar, hardlink_or_copy, image_hash

CLASS_NAME = "plate"
ATTRIBUTE_NAME = "plate number"
DATASET_ID = "piotrstefaskiue/poland-vehicle-license-plate-dataset"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}  # matched case-insensitively
SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")
SPLIT_THRESHOLD_TRAIN = 60
SPLIT_THRESHOLD_VAL = 70


class PreparationError(RuntimeError):
    """Data problem; preparation must stop, never silently skip."""


@dataclass(frozen=True)
class Sample:
    image_id: str
    image_path: Path
    width: int
    height: int
    boxes: tuple[tuple[float, float, float, float], ...]
    plate_texts: tuple[str, ...]  # aligned with boxes
    dhash: int
    digest: str

    def yolo_lines(self) -> list[str]:
        lines = []
        for xtl, ytl, xbr, ybr in self.boxes:
            cx = ((xtl + xbr) / 2.0) / self.width
            cy = ((ytl + ybr) / 2.0) / self.height
            bw = (xbr - xtl) / self.width
            bh = (ybr - ytl) / self.height
            lines.append(f"0 {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}")
        return lines


def discover(raw_dir: Path) -> tuple[Path, dict[str, Path], str]:
    """Locate annotations.xml, photo files and best-known dataset version."""
    root = Path(raw_dir)
    if not root.is_dir():
        raise PreparationError(f"Brak katalogu wejściowego: {root}")
    xml_path = root / "annotations.xml"
    if not xml_path.is_file():
        raise PreparationError(f"Brak pliku annotations.xml: {xml_path}")
    photos: dict[str, Path] = {}
    basenames: dict[str, Path] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        rel = path.relative_to(root).as_posix()
        name = Path(rel).name
        if name in basenames:
            raise PreparationError(f"Kolizja nazw plików w katalogu wejściowym: {name}")
        photos[rel] = path
        basenames[name] = path
        photos[name] = path
    if not photos:
        raise PreparationError("Brak zdjęć w katalogu wejściowym")
    version = ""
    match = re.search(r"versions/(\d+)", str(root))
    if match:
        version = match.group(1)
    return xml_path, photos, version


def parse_cvat(xml_path: Path, photos: dict[str, Path]) -> tuple[list[Sample], list[str]]:
    """Strictly parse CVAT XML; returns (labeled samples, unannotated names)."""
    samples: list[Sample] = []
    unannotated: list[str] = []
    seen_stems: set[str] = set()
    for image in ET.parse(xml_path).getroot().findall("image"):
        name = (image.get("name") or "").strip()
        if (
            not name
            or not SAFE_NAME.match(name)
            or "/" in name
            or "\\" in name
            or ".." in name
        ):
            raise PreparationError(f"Niepoprawna nazwa obrazu w XML: {name!r}")
        try:
            width = int(float(image.get("width")))
            height = int(float(image.get("height")))
        except (TypeError, ValueError) as exc:
            raise PreparationError(f"Obraz {name}: brak poprawnych wymiarów w XML") from exc
        if width < 2 or height < 2:
            raise PreparationError(f"Obraz {name}: wymiary w XML {width}x{height} są za małe")
        image_path = photos.get(name)
        if image_path is None:
            raise PreparationError(f"Brak zdjęcia dla adnotacji XML: {name}")
        try:
            with Image.open(image_path) as img:
                actual_w, actual_h = img.size
                img.verify()
        except (UnidentifiedImageError, OSError) as exc:
            raise PreparationError(f"Niepoprawny plik obrazu: {name}") from exc
        if (actual_w, actual_h) != (width, height):
            raise PreparationError(
                f"Obraz {name}: wymiary XML {width}x{height} "
                f"różnią się od rzeczywistych {actual_w}x{actual_h}"
            )
        stem = Path(name).stem.lower()
        if stem in seen_stems:
            raise PreparationError(f"Kolizja stemów nazw: {name}")
        seen_stems.add(stem)
        boxes: list[tuple[float, float, float, float]] = []
        plate_texts: list[str] = []
        for box in image.findall("box"):
            if box.get("label") != CLASS_NAME:
                raise PreparationError(
                    f"Obraz {name}: oczekiwana adnotacja '{CLASS_NAME}', "
                    f"znaleziono '{box.get('label')}'"
                )
            try:
                xtl, ytl, xbr, ybr = (float(box.get(k)) for k in ("xtl", "ytl", "xbr", "ybr"))
            except (TypeError, ValueError) as exc:
                raise PreparationError(
                    f"Obraz {name}: adnotacja bez poprawnych współrzędnych"
                ) from exc
            if not (
                0.0 <= xtl <= xbr <= float(width)
                and 0.0 <= ytl <= ybr <= float(height)
                and xbr - xtl >= 1.0
                and ybr - ytl >= 1.0
            ):
                raise PreparationError(
                    f"Obraz {name}: box poza obrazem lub zdegenerowany: "
                    f"({xtl}, {ytl}, {xbr}, {ybr}) na {width}x{height}"
                )
            boxes.append((xtl, ytl, xbr, ybr))
            texts = [
                (attr.text or "").strip()
                for attr in box.findall("attribute")
                if attr.get("name") == ATTRIBUTE_NAME
            ]
            plate_texts.append(" | ".join(texts))
        if not boxes:
            unannotated.append(name)
        dhash, digest = image_hash(image_path)
        samples.append(
            Sample(name, image_path, width, height, tuple(boxes), tuple(plate_texts), dhash, digest)
        )
    return samples, unannotated


def assign_splits(samples: list[Sample], groups: list[list[int]], seed: int) -> dict[str, str]:
    """Deterministic 60/10/30 split of whole similar-image groups."""
    assignments: dict[str, str] = {}
    for group in groups:
        token = min(samples[idx].image_id for idx in group)
        bucket = int(hashlib.sha256(f"{seed}:{token}".encode()).hexdigest(), 16) % 100
        split = (
            "train"
            if bucket < SPLIT_THRESHOLD_TRAIN
            else "val"
            if bucket < SPLIT_THRESHOLD_VAL
            else "test"
        )
        for idx in group:
            assignments[samples[idx].image_id] = split
    return assignments


def yaml_quote(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)


def build(
    out_dir: Path,
    samples: list[Sample],
    assignments: dict[str, str],
    unannotated: list[str],
    seed: int,
    raw_dir: Path,
    version: str,
) -> dict[str, object]:
    out_dir = Path(out_dir).resolve()
    raw_dir = Path(raw_dir).resolve()
    if out_dir == raw_dir:
        raise PreparationError("Katalog wyjściowy nie może być katalogiem wejściowym")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise PreparationError(
            f"Katalog wyjściowy już istnieje i jest niepusty: {out_dir} — nie nadpisuję"
        )
    counts = {"train": 0, "val": 0, "test": 0}
    for item in samples:
        if item.boxes:
            counts[assignments[item.image_id]] += 1
    if min(counts.values()) == 0:
        raise PreparationError(f"Pusty split przed zapisaniem: {counts}")
    for split in ("train", "val", "test"):
        (out_dir / "images" / split).mkdir(parents=True)
        (out_dir / "labels" / split).mkdir(parents=True)
    manifest: list[dict[str, object]] = []
    with (out_dir / "gt.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["image", "width", "height", "xmin", "ymin", "xmax", "ymax", "plate_text"]
        )
        for item in sorted(samples, key=lambda s: s.image_id):
            if item.boxes:
                split = assignments[item.image_id]
                image_target = out_dir / "images" / split / item.image_id
                label_target = out_dir / "labels" / split / f"{Path(item.image_id).stem}.txt"
                hardlink_or_copy(item.image_path, image_target)
                label_target.write_text("\n".join(item.yolo_lines()) + "\n", encoding="utf-8")
                for (xtl, ytl, xbr, ybr), text in zip(item.boxes, item.plate_texts):
                    writer.writerow([item.image_id, item.width, item.height, xtl, ytl, xbr, ybr, text])
            manifest.append(
                {
                    "id": Path(item.image_id).stem,
                    "image": item.image_id,
                    "source": f"{DATASET_ID}" + (f"#versions/{version}" if version else ""),
                    "sha256": item.digest,
                    "dhash": f"{item.dhash:018x}",
                    "split": assignments[item.image_id] if item.boxes else None,
                    "boxes": len(item.boxes),
                    "plate_texts": list(item.plate_texts),
                    "labeled": bool(item.boxes),
                }
            )
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {"dataset": DATASET_ID, "seed": seed, "images": manifest},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    root_path = yaml_quote(str(out_dir))
    (out_dir / "dataset.yaml").write_text(
        f"train: images/train\nval: images/val\ntest: images/test\nnames:\n  0: plate\n"
        f"path: {root_path}\n",
        encoding="utf-8",
    )
    (out_dir / "SOURCE.md").write_text(
        "# Dataset provenance\n\n"
        f"Kaggle dataset: {DATASET_ID}"
        + (f" (version {version})\n" if version else "\n")
        + "\n"
        "Pulled with kagglehub (KAGGLEHUB_CACHE) into a local cache; raw files "
        "stay on /storage and are not committed.\n\n"
        "Kaggle tags this dataset's license as **Other** and its description "
        "states no explicit use conditions. No agreement to training, "
        "redistribution or public exposure is assumed; the data is used only "
        "locally for training and evaluation in this repository.\n\n"
        "Annotations are CVAT XML with label `plate` and text attribute "
        "`plate number`. The real texts are preserved verbatim in `gt.csv`; "
        "no OCR model is implemented at this stage. Image IDs, hashes and split "
        "assignments are retained in `manifest.json`. Images without boxes "
        "(if any) are kept in the manifest as unlabeled and excluded from "
        "the YOLO splits.\n",
        encoding="utf-8",
    )
    text_labels = sum(1 for item in samples for text in item.plate_texts if text)
    return {
        "counts": counts,
        "text_labels": text_labels,
        "unannotated": list(unannotated),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    xml_path, photos, version = discover(args.raw_dir)
    samples, unannotated = parse_cvat(xml_path, photos)
    groups = group_similar(samples)
    assignments = assign_splits(samples, groups, args.seed)
    summary = build(args.out_dir, samples, assignments, unannotated, args.seed, args.raw_dir, version)
    print(
        f"[prepared] images={len(samples)} boxes={sum(len(s.boxes) for s in samples)} "
        f"splits={summary['counts']} text_labels={summary['text_labels']} "
        f"unannotated={len(summary['unannotated'])}",
        flush=True,
    )
    print(f"[dedupe] {len(groups)} perceptual groups", flush=True)


if __name__ == "__main__":
    main()
