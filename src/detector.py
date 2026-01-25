from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
try:
    # Runtime: to jest publiczne API Ultralytics, ale Pylance/pyright czasem krzyczy na stuby.
    from ultralytics import YOLO as UltralyticsYOLO  # type: ignore[reportPrivateImportUsage]
except ImportError as e:
    raise ImportError(
        "Brak paczki 'ultralytics'. Zainstaluj: python -m pip install ultralytics"
    ) from e


@dataclass(frozen=True)
class Detection:
    bbox_xyxy: Tuple[float, float, float, float]
    conf: float
    cls: int


class YoloPlateDetector:
    def __init__(
        self,
        weights_path: str,
        img_size: int = 640,
        conf: float = 0.25,
        device: str = "auto",          # "cpu", "cuda", "mps", "auto"
        fallback_to_cpu: bool = True,
        max_det: int = 1,
    ):
        self.model = UltralyticsYOLO(weights_path)
        self.img_size = int(img_size)
        self.conf = float(conf)
        self.device = device
        self.fallback_to_cpu = bool(fallback_to_cpu)
        self.max_det = int(max_det)

        # mały „warmup” pomaga na niektórych konfiguracjach
        # (szczególnie gdy pierwsze wywołanie ma inny rozmiar obrazu)
        try:
            _ = self.model.predict(
                source=np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8),
                imgsz=self.img_size,
                conf=self.conf,
                device=self.device,
                verbose=False,
                max_det=self.max_det,
            )
        except Exception:
            # warmup nie jest krytyczny
            pass

    @staticmethod
    def _sanitize_input(img_or_path: Union[np.ndarray, str]) -> Union[np.ndarray, str]:
        if isinstance(img_or_path, str):
            return img_or_path

        img = img_or_path
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]

        # Ultralytics najlepiej lubi uint8
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        # contiguous (czasem eliminuje dziwne crashe / błędy backendu)
        if not img.flags["C_CONTIGUOUS"]:
            img = np.ascontiguousarray(img)

        return img

    def _predict(self, source: Union[np.ndarray, str], device: str):
        return self.model.predict(
            source=source,
            imgsz=self.img_size,
            conf=self.conf,
            device=device,
            verbose=False,
            agnostic_nms=True,
            max_det=self.max_det,
        )

    def detect_one(self, img_or_path: Union[np.ndarray, str]) -> Optional[Detection]:
        source = self._sanitize_input(img_or_path)

        try:
            results = self._predict(source, self.device)
        except Exception as e:
            if not self.fallback_to_cpu:
                return None
            # fallback na CPU, jeśli GPU/MPS się wywali
            try:
                results = self._predict(source, "cpu")
            except Exception:
                return None

        if not results:
            return None

        r = results[0]
        boxes = getattr(r, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return None

        # wybierz najlepszy box po conf
        confs = boxes.conf.detach().cpu().numpy()
        best_i = int(np.argmax(confs))

        xyxy = boxes.xyxy[best_i].detach().cpu().numpy().tolist()
        cls = int(boxes.cls[best_i].detach().cpu().item())
        conf = float(boxes.conf[best_i].detach().cpu().item())

        x1, y1, x2, y2 = map(float, xyxy)
        return Detection((x1, y1, x2, y2), conf=conf, cls=cls)