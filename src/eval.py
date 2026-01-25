from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import time
import cv2
import numpy as np
from tqdm import tqdm

from .ocr import ocr_ensemble_plate
from .metrics import iou_xyxy, normalize_plate, calculate_final_grade

@dataclass
class GroundTruth:
    bbox_xyxy: Tuple[float, float, float, float]
    plate_text: str

def crop_safe(img: np.ndarray, xyxy: Tuple[float,float,float,float]) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    x1 = max(0, min(w-1, x1))
    y1 = max(0, min(h-1, y1))
    x2 = max(0, min(w,   x2))
    y2 = max(0, min(h,   y2))
    if x2 <= x1 or y2 <= y1:
        return img[0:1, 0:1].copy()
    return img[y1:y2, x1:x2].copy()

def pad_xyxy(xyxy: Tuple[float, float, float, float], pad: float = 0.12) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = xyxy
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    dx = bw * pad
    dy = bh * pad
    return (x1 - dx, y1 - dy, x2 + dx, y2 + dy)

def evaluate(
    image_paths: List[str],
    gt_map: Dict[str, GroundTruth],
    detector_weights: str,
    limit_100: bool = True,
    ocr_engine: str = "default",
    lprnet_weights: Optional[str] = None,
    ocr_device: str = "auto",
    **_ignored: object,
) -> Dict[str, float]:

    if limit_100:
        image_paths = image_paths[:100]

    from .detector import YoloPlateDetector
    detector = YoloPlateDetector(detector_weights, img_size=640, conf=0.25)

    correct = 0
    total = 0
    ious: List[float] = []

    t0 = time.perf_counter()

    for p in tqdm(image_paths, desc="Eval"):
        img = cv2.imread(p)
        if img is None:
            continue

        key = p
        if key not in gt_map:
            # Fallback dla nazw plików
            import os
            key = os.path.basename(p)
            if key not in gt_map:
                continue

        gt = gt_map[key]
        det = detector.detect_one(img)

        total += 1

        if det is None:
            ious.append(0.0)
            continue

        ious.append(iou_xyxy(det.bbox_xyxy, gt.bbox_xyxy))

        # 1. Wycinanie z marginesem
        plate_crop = crop_safe(img, pad_xyxy(det.bbox_xyxy, pad=0.05))
        
        # USUNIĘTO cv2.resize! Przekazujemy surowy crop do OCR.
        plate_crop = np.ascontiguousarray(plate_crop, dtype=np.uint8)

        # 2. OCR
        if (ocr_engine or "").lower() in ("lprnet", "lpr"):
            # LPRNet jest opcjonalny – próbujemy pobrać funkcję dynamicznie
            from . import ocr as ocr_mod

            fn = getattr(ocr_mod, "ocr_lprnet_plate", None)
            if fn is None:
                raise ImportError(
                    "Wybrano ocr_engine='lprnet', ale funkcja ocr_lprnet_plate nie istnieje w src/ocr.py. "
                    "Zmień --ocr-engine na default albo dodaj implementację LPRNet."
                )
            pred_txt = fn(
                plate_crop,
                weights_path=lprnet_weights,
                device=ocr_device,
            )
        else:
            pred_txt = ocr_ensemble_plate(plate_crop)

        # 3. Weryfikacja
        if normalize_plate(pred_txt) == normalize_plate(gt.plate_text):
            correct += 1
        else:
            # Opcjonalnie: Debug print (odkomentuj jeśli chcesz widzieć błędy)
            print(f"MISMATCH: {gt.plate_text} vs {pred_txt}")
            pass

    t1 = time.perf_counter()
    elapsed = t1 - t0

    acc = (correct / total * 100.0) if total > 0 else 0.0
    mean_iou = float(np.mean(ious)) if ious else 0.0
    grade = calculate_final_grade(acc, elapsed)

    return {
        "tested": float(total),
        "accuracy_percent": float(acc),
        "processing_time_sec_100": float(elapsed),
        "mean_iou": float(mean_iou),
        "final_grade": float(grade),
    }