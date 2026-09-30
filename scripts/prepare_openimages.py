#!/usr/bin/env python3
"""Download a bounded Open Images plate subset and build leak-aware YOLO splits.

Only the annotated ``train`` and ``validation`` trees are used. The unrelated
``harvested_plates`` directory in the mirror is deliberately never enumerated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

REPO = "shravya11/vehicle-registration-plate"
REVISION = "main"
CLASS_DIR = "Vehicle registration plate"
USER_AGENT = "plates-dataset-preparer/1.0"


@dataclass(frozen=True)
class Sample:
    source_split: str
    image_id: str
    image_path: Path
    label_path: Path
    dhash: int
    digest: str


def request_json(url: str) -> tuple[list[dict[str, object]], str | None]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as response:
        payload = json.load(response)
        link = response.headers.get("Link")
    next_url = None
    if link:
        for part in link.split(","):
            if 'rel="next"' in part:
                next_url = part.split(";", 1)[0].strip().strip("<>")
    return payload, next_url


def list_files(split: str, child: str = "") -> list[dict[str, object]]:
    path = f"{split}/{CLASS_DIR}" + (f"/{child}" if child else "")
    encoded = urllib.parse.quote(path, safe="/")
    url = (
        f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/{encoded}"
        "?recursive=false&expand=false&limit=1000"
    )
    found: list[dict[str, object]] = []
    while url:
        page, url = request_json(url)
        found.extend(item for item in page if item.get("type") == "file")
    return found


def stable_pick(items: list[dict[str, object]], limit: int, seed: str) -> list[dict[str, object]]:
    def key(item: dict[str, object]) -> str:
        return hashlib.sha256(f"{seed}:{item['path']}".encode()).hexdigest()

    return sorted(items, key=key)[:limit]


def download(url: str, destination: Path, retries: int = 4) -> None:
    if destination.is_file() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=90) as response, temporary.open("wb") as out:
                shutil.copyfileobj(response, out, length=256 * 1024)
            os.replace(temporary, destination)
            return
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt + 1 == retries:
                raise
            time.sleep(2 ** attempt)


def resolve_url(path: str) -> str:
    return (
        f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/"
        f"{urllib.parse.quote(path, safe='/')}?download=true"
    )


def image_hash(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("L").resize((9, 8))
        # Pillow 14 removes getdata(); keep compatibility with older releases.
        get_pixels = getattr(image, "get_flattened_data", image.getdata)
        pixels = list(get_pixels())
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | int(pixels[row * 9 + col] > pixels[row * 9 + col + 1])
    return bits, digest


def parse_boxes(label_path: Path, width: int, height: int) -> list[str]:
    output: list[str] = []
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw.rsplit(maxsplit=4)
        if len(parts) != 5 or parts[0] != CLASS_DIR:
            continue
        x1, y1, x2, y2 = map(float, parts[1:])
        x1, x2 = sorted((max(0.0, min(width, x1)), max(0.0, min(width, x2))))
        y1, y2 = sorted((max(0.0, min(height, y1)), max(0.0, min(height, y2))))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        output.append(
            f"0 {((x1+x2)/2)/width:.8f} {((y1+y2)/2)/height:.8f} "
            f"{(x2-x1)/width:.8f} {(y2-y1)/height:.8f}"
        )
    return output


def collect(root: Path, train_limit: int, validation_limit: int) -> list[Sample]:
    samples: list[Sample] = []
    for split, limit in (("train", train_limit), ("validation", validation_limit)):
        images = [x for x in list_files(split) if str(x["path"]).lower().endswith((".jpg", ".jpeg", ".png"))]
        labels = {Path(str(x["path"])).stem: x for x in list_files(split, "Label")}
        selected = [x for x in stable_pick(images, limit, "plates-2026") if Path(str(x["path"])).stem in labels]
        if len(selected) < limit:
            raise RuntimeError(f"Only {len(selected)} paired samples available for {split}")
        for number, image_meta in enumerate(selected, 1):
            image_id = Path(str(image_meta["path"])).stem
            image_path = root / "raw" / split / "images" / Path(str(image_meta["path"])).name
            label_meta = labels[image_id]
            label_path = root / "raw" / split / "labels" / Path(str(label_meta["path"])).name
            download(resolve_url(str(image_meta["path"])), image_path)
            download(resolve_url(str(label_meta["path"])), label_path)
            try:
                dhash, digest = image_hash(image_path)
                with Image.open(image_path) as image:
                    image.verify()
            except (UnidentifiedImageError, OSError) as exc:
                image_path.unlink(missing_ok=True)
                raise RuntimeError(f"Invalid downloaded image {image_id}") from exc
            samples.append(Sample(split, image_id, image_path, label_path, dhash, digest))
            if number % 50 == 0:
                print(f"[download] {split}: {number}/{len(selected)}", flush=True)
    return samples


def group_similar(samples: list[Sample], max_distance: int = 3) -> list[list[int]]:
    parent = list(range(len(samples)))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[b] = a

    exact: dict[str, int] = {}
    for idx, sample in enumerate(samples):
        if sample.digest in exact:
            union(idx, exact[sample.digest])
        else:
            exact[sample.digest] = idx
    for left in range(len(samples)):
        for right in range(left + 1, len(samples)):
            if (samples[left].dhash ^ samples[right].dhash).bit_count() <= max_distance:
                union(left, right)
    groups: dict[int, list[int]] = {}
    for idx in range(len(samples)):
        groups.setdefault(find(idx), []).append(idx)
    return list(groups.values())


def assign_splits(samples: list[Sample], groups: list[list[int]]) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for group in groups:
        members = [samples[idx] for idx in group]
        if any(item.source_split == "train" for item in members):
            split = "train"
        else:
            token = min(item.image_id for item in members)
            split = "val" if int(hashlib.sha256(token.encode()).hexdigest(), 16) % 2 == 0 else "test"
        for item in members:
            assignments[f"{item.source_split}/{item.image_id}"] = split
    return assignments


def hardlink_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.unlink(missing_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def build(root: Path, samples: list[Sample], assignments: dict[str, str]) -> None:
    prepared = root / "prepared"
    for split in ("train", "val", "test"):
        (prepared / "images" / split).mkdir(parents=True, exist_ok=True)
        (prepared / "labels" / split).mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    counts = {"train": 0, "val": 0, "test": 0}
    for item in samples:
        split = assignments[f"{item.source_split}/{item.image_id}"]
        with Image.open(item.image_path) as image:
            width, height = image.size
        boxes = parse_boxes(item.label_path, width, height)
        if not boxes:
            continue
        image_target = prepared / "images" / split / item.image_path.name
        label_target = prepared / "labels" / split / f"{item.image_id}.txt"
        hardlink_or_copy(item.image_path, image_target)
        label_target.write_text("\n".join(boxes) + "\n", encoding="utf-8")
        counts[split] += 1
        manifest.append({"id": item.image_id, "source_split": item.source_split, "split": split, "boxes": len(boxes), "sha256": item.digest})
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (root / "dataset.yaml").write_text(
        f"path: {prepared}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: plate\n",
        encoding="utf-8",
    )
    (root / "SOURCE.md").write_text(
        "# Dataset provenance\n\n"
        "Selected images are an Open Images plate subset mirrored at "
        "https://huggingface.co/datasets/shravya11/vehicle-registration-plate.\n\n"
        "Open Images annotations are CC BY 4.0; source images are listed by Open Images "
        "as CC BY 2.0, with no warranty from Open Images about each image's individual "
        "license status. The mirror's dataset card declares CC BY 4.0. The "
        "`harvested_plates` tree was not used. Image IDs, hashes and "
        "split assignments are retained in `manifest.json`.\n",
        encoding="utf-8",
    )
    print(f"[prepared] {counts}", flush=True)
    if min(counts.values()) == 0:
        raise RuntimeError(f"A dataset split is empty: {counts}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=600)
    parser.add_argument("--validation-limit", type=int, default=240)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    samples = collect(args.root, args.train_limit, args.validation_limit)
    groups = group_similar(samples)
    assignments = assign_splits(samples, groups)
    build(args.root, samples, assignments)
    print(f"[dedupe] {len(samples)} images in {len(groups)} perceptual groups", flush=True)


if __name__ == "__main__":
    main()
