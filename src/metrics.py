from __future__ import annotations
import re

def iou_xyxy(a, b) -> float:
    """
    a,b: (x1,y1,x2,y2)
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area

    return float(inter_area / union) if union > 0 else 0.0

PLATE_CLEAN_RE = re.compile(r"[^A-Z0-9]")

def normalize_plate(s: str) -> str:
    s = (s or "").upper().strip()
    s = PLATE_CLEAN_RE.sub("", s)

    # lekkie heurystyki pod typowe pomyłki OCR
    # (nie zawsze warto – ale często podbija accuracy)
    s = s.replace("O", "0") if any(ch.isdigit() for ch in s) else s
    s = s.replace("I", "1") if any(ch.isdigit() for ch in s) else s
    return s

def calculate_final_grade(accuracy_percent: float, processing_time_sec: float) -> float:
    """
    Skala 2.0–5.0, zaokrąglenie do 0.5.
    Minimalne wymagania: accuracy>=60 i czas<=60.
    Źródło: treść projektu.  [oai_citation:5‡0. Projekt - detekcja oaz OCR tablic rejestracyjnych.pdf](file-service://file-SP7RSHL3kDJdjjLj3syLeu)
    """
    if accuracy_percent < 60 or processing_time_sec > 60:
        return 2.0

    accuracy_norm = (accuracy_percent - 60) / 40
    time_norm = (60 - processing_time_sec) / 50

    score = 0.7 * accuracy_norm + 0.3 * time_norm
    grade = 2.0 + 3.0 * score
    return round(grade * 2) / 2