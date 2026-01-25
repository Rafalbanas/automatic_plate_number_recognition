from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Dict, Optional
import sys
import xml.etree.ElementTree as ET
import cv2

from src.eval import evaluate, GroundTruth
from src.metrics import normalize_plate


def _clean_plate_text(s: str) -> str:
    # Ujednolicona normalizacja: tylko A-Z0-9, uppercase
    return normalize_plate(s)


def _pick_best_box(box_elems) -> Optional[ET.Element]:
    """
    Jeśli zdarzy się kilka boxów na 1 obraz, wybierz "największy" (pole powierzchni).
    """
    best = None
    best_area = -1.0
    for b in box_elems:
        try:
            xtl = float(b.attrib["xtl"])
            ytl = float(b.attrib["ytl"])
            xbr = float(b.attrib["xbr"])
            ybr = float(b.attrib["ybr"])
            area = max(0.0, xbr - xtl) * max(0.0, ybr - ytl)
        except Exception:
            continue

        if area > best_area:
            best_area = area
            best = b
    return best


def parse_cvat_annotations_xml(annotations_xml_path: str) -> Dict[str, GroundTruth]:
    """
    Parsuje CVAT annotations.xml i zwraca mapę:
      {basename_obrazu: GroundTruth((x1,y1,x2,y2), plate_text)}
    """
    tree = ET.parse(annotations_xml_path)
    root = tree.getroot()

    gt_by_basename: Dict[str, GroundTruth] = {}

    for img in root.findall("image"):
        name = img.attrib.get("name")
        if not name:
            continue

        boxes = [b for b in img.findall("box") if b.attrib.get("label") == "plate"]
        if not boxes:
            continue

        box = _pick_best_box(boxes)
        if box is None:
            continue

        try:
            x1 = float(box.attrib["xtl"])
            y1 = float(box.attrib["ytl"])
            x2 = float(box.attrib["xbr"])
            y2 = float(box.attrib["ybr"])
        except KeyError:
            continue

        plate_text = ""
        for attr in box.findall("attribute"):
            if attr.attrib.get("name") == "plate number":
                plate_text = (attr.text or "").strip()
                break

        if not plate_text:
            continue

        plate_text = _clean_plate_text(plate_text)
        gt_by_basename[os.path.basename(name)] = GroundTruth((x1, y1, x2, y2), plate_text)

    return gt_by_basename


def build_gt_map_for_images(
    image_paths: list[str],
    annotations_xml_path: str,
) -> Dict[str, GroundTruth]:
    """
    Mapuje GT z basenamów do konkretnych ścieżek obrazów.
    Zwraca:
      {image_path: GroundTruth(...)}
    """
    gt_by_basename = parse_cvat_annotations_xml(annotations_xml_path)

    gt_map: Dict[str, GroundTruth] = {}
    missing = 0

    for p in image_paths:
        bn = os.path.basename(p)
        gt = gt_by_basename.get(bn)
        if gt is None:
            missing += 1
            continue
        gt_map[p] = gt

    print(f"[GT] images={len(image_paths)}  matched={len(gt_map)}  missing={missing}")
    return gt_map


def main() -> None:
    print("Starting evaluation script...", flush=True)
    argp = argparse.ArgumentParser(description="Evaluate plate detector + OCR on 100 images")

    argp.add_argument(
        "--annotations",
        type=str,
        default="data/raw/annotations.xml",
        help="Path to CVAT annotations.xml (default: data/raw/annotations.xml)",
    )

    argp.add_argument(
        "--weights",
        type=str,
        default="runs/detect/plate_det/yolov8n_100e_l49/weights/best.pt",
        help="Path to YOLO weights .pt (default: runs/detect/plate_det/yolov8n_100e_l49/weights/best.pt)",
    )
    argp.add_argument(
        "--ocr",
        type=str,
        choices=["default", "lprnet"],
        default="default",
        help="OCR backend: default (current pipeline) or lprnet",
    )
    argp.add_argument(
        "--lprnet-weights",
        type=str,
        default="LPRNet_Pytorch/weights/Final_LPRNet_model.pth",
        help="Path to LPRNet .pth weights",
    )
    argp.add_argument(
        "--ocr-device",
        type=str,
        choices=["cpu", "mps", "cuda"],
        default="cpu",
        help="Device for OCR model (lprnet): cpu/mps/cuda",
    )

    args = argp.parse_args()

    print("Searching for images in data/raw/**/*.* ...", flush=True)
    images = sorted(glob.glob("data/raw/**/*.*", recursive=True))
    images = [p for p in images if os.path.isfile(p) and p.lower().endswith((".jpg", ".jpeg", ".png"))]
    print(f"Found {len(images)} candidate image files.", flush=True)

    annotations_xml_path = Path(args.annotations)
    if not annotations_xml_path.exists():
        print(f"Error: Annotations file not found at {annotations_xml_path}", file=sys.stderr)
        sys.exit(1)

    gt_map = build_gt_map_for_images(images, str(annotations_xml_path))
    images_eval = sorted(gt_map.keys())

    if not images_eval:
        print("No images matched with annotations. Exiting.", flush=True)
        return

    print(f"Starting evaluation on {len(images_eval)} images using weights: {args.weights}", flush=True)
    print("Loading heavy libraries (PyTorch, Ultralytics)... this may take a moment on Google Drive.", flush=True)
    res = evaluate(
        image_paths=images_eval,
        gt_map=gt_map,
        detector_weights=args.weights,
        limit_100=True,
        ocr_engine=args.ocr,
        lprnet_weights=args.lprnet_weights,
        ocr_device=args.ocr_device,
    )

    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()