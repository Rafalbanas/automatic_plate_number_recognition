from __future__ import annotations

import io
import os
import re
import threading
import time
from dataclasses import dataclass
from statistics import mean
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_FILE_BYTES = int(os.getenv("PLATES_MAX_FILE_BYTES", str(4 * 1024 * 1024)))
MAX_PIXELS = int(os.getenv("PLATES_MAX_PIXELS", "12000000"))
MAX_DIMENSION = int(os.getenv("PLATES_MAX_DIMENSION", "4096"))
DETECTOR_MODEL = os.getenv(
    "PLATES_DETECTOR_MODEL", "yolo-v9-t-384-license-plate-end2end"
)
OCR_MODEL = os.getenv("PLATES_OCR_MODEL", "cct-xs-v2-global-model")
DETECTION_CONFIDENCE = float(os.getenv("PLATES_DETECTION_CONFIDENCE", "0.35"))
MIN_OCR_CONFIDENCE = float(os.getenv("PLATES_MIN_OCR_CONFIDENCE", "0.45"))
PLATE_RE = re.compile(r"^[A-Z0-9]{4,10}$")

Image.MAX_IMAGE_PIXELS = MAX_PIXELS


class InputError(ValueError):
    """Safe validation error which can be returned to a client."""


@dataclass(frozen=True)
class Analysis:
    plates: list[dict[str, Any]]
    width: int
    height: int
    elapsed_ms: int


def decode_image(payload: bytes, content_type: str) -> tuple[np.ndarray, int, int]:
    if not payload:
        raise InputError("empty")
    if len(payload) > MAX_FILE_BYTES:
        raise InputError("file_too_large")
    if content_type.split(";", 1)[0].strip().lower() not in {
        "image/jpeg",
        "image/png",
        "image/webp",
    }:
        raise InputError("unsupported_type")

    try:
        with Image.open(io.BytesIO(payload)) as probe:
            probe.verify()
        with Image.open(io.BytesIO(payload)) as image:
            if getattr(image, "n_frames", 1) != 1:
                raise InputError("animated_image")
            fmt = (image.format or "").upper()
            if fmt not in {"JPEG", "PNG", "WEBP"}:
                raise InputError("unsupported_content")
            width, height = image.size
            if width < 32 or height < 32:
                raise InputError("image_too_small")
            if width > MAX_DIMENSION or height > MAX_DIMENSION:
                raise InputError("dimensions_too_large")
            if width * height > MAX_PIXELS:
                raise InputError("too_many_pixels")
            image = ImageOps.exif_transpose(image).convert("RGB")
            rgb = np.asarray(image, dtype=np.uint8)
    except InputError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise InputError("invalid_image") from exc

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return np.ascontiguousarray(bgr), int(rgb.shape[1]), int(rgb.shape[0])


class PlateEngine:
    """One process-wide model pair; calls are serialized to cap memory use."""

    def __init__(self) -> None:
        from fast_alpr import ALPR

        self._lock = threading.Lock()
        self._model = ALPR(
            detector_model=DETECTOR_MODEL,
            ocr_model=OCR_MODEL,
            detector_conf_thresh=DETECTION_CONFIDENCE,
        )
        # Warm both ONNX sessions without writing any user data.
        self._model.predict(np.zeros((384, 384, 3), dtype=np.uint8))

    @staticmethod
    def _ocr_confidence(value: Any) -> float | None:
        if isinstance(value, (float, int)):
            return float(value)
        if isinstance(value, (list, tuple)) and value:
            numeric = [float(item) for item in value if isinstance(item, (float, int))]
            return mean(numeric) if numeric else None
        return None

    def analyze(self, payload: bytes, content_type: str) -> Analysis:
        frame, width, height = decode_image(payload, content_type)
        started = time.perf_counter()
        with self._lock:
            results = self._model.predict(frame)
        elapsed_ms = round((time.perf_counter() - started) * 1000)

        plates: list[dict[str, Any]] = []
        for item in results[:3]:
            bbox = item.detection.bounding_box
            x1 = max(0, min(width, int(round(bbox.x1))))
            y1 = max(0, min(height, int(round(bbox.y1))))
            x2 = max(0, min(width, int(round(bbox.x2))))
            y2 = max(0, min(height, int(round(bbox.y2))))
            if x2 <= x1 or y2 <= y1:
                continue

            text = ""
            accepted = False
            if item.ocr is not None:
                text = re.sub(r"[^A-Z0-9]", "", (item.ocr.text or "").upper())
                confidence = self._ocr_confidence(item.ocr.confidence)
                accepted = bool(
                    PLATE_RE.fullmatch(text)
                    and confidence is not None
                    and confidence >= MIN_OCR_CONFIDENCE
                )
            plates.append(
                {
                    "box": [x1, y1, x2, y2],
                    "text": text if accepted else "",
                    "read": accepted,
                }
            )

        return Analysis(
            plates=plates,
            width=width,
            height=height,
            elapsed_ms=elapsed_ms,
        )
