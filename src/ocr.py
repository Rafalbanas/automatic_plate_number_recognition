import numpy as np
import cv2
import re
import os
from pathlib import Path
from typing import Any, Literal, cast

# Set to False to silence OCR logs.
DEBUG_LOGS = True


def _clean_text(text: str) -> str:
    """Cleans the OCR result by removing unwanted characters and formatting.
    Keeps only uppercase letters and digits.
    """
    if not text:
        return ""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def _score_text(text: str) -> int:
    """Prosta heurystyka oceny jakości odczytu na podstawie długości.
    Polskie tablice mają zazwyczaj 7-8 znaków.
    """
    if not text:
        return 0
    length = len(text)
    if 7 <= length <= 8:
        return 100
    if 4 <= length <= 6:
        return 80
    return 40


def _preprocess_tess(plate_bgr: np.ndarray) -> np.ndarray:
    """Przygotowanie obrazu dla Tesseracta (skala szarości, binaryzacja)."""
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    # Powiększenie obrazu pomaga Tesseractowi przy małych tablicach
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    # Adaptive threshold jest znacznie lepszy dla tablic niż Otsu
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return thresh


_MODEL_CACHE = None
_MODEL_LOAD_ATTEMPTED = False


class _FastPlateOCRWrapper:
    """Thin wrapper that normalizes fast-plate-ocr v1.x API to `recognize_batch(images)->list[str]`."""

    def __init__(
        self,
        hub_model: str | None = None,
        onnx_model_path: str | None = None,
        plate_config_path: str | None = None,
        device: Literal["cuda", "cpu", "auto"] = "auto",
    ):
        # fast-plate-ocr performs ONNX inference via LicensePlateRecognizer
        from fast_plate_ocr import LicensePlateRecognizer

        if onnx_model_path and plate_config_path:
            self.recognizer = LicensePlateRecognizer(
                device=cast(Any, device),
                onnx_model_path=onnx_model_path,
                plate_config_path=plate_config_path,
            )
        else:
            if not hub_model:
                raise ValueError(
                    "hub_model is required when no custom onnx/config is provided"
                )
            self.recognizer = LicensePlateRecognizer(
                hub_ocr_model=cast(Any, hub_model),
                device=cast(Any, device),
            )

    def recognize_batch(self, images: list[np.ndarray]) -> list[str]:
        """Run FastPlate OCR on a batch of images.

        FastPlate hub models are not consistent across versions/exports:
        - some expect NHWC with 1 channel (grayscale)
        - some expect NHWC with 3 channels (RGB)

        This method tries multiple safe representations in a fixed order.
        """
        if not images:
            return []

        def _ensure_uint8_contiguous(arr: np.ndarray) -> np.ndarray:
            if arr is None or getattr(arr, "size", 0) == 0:
                return arr

            if arr.dtype != np.uint8:
                # If we get float images (often 0..1), scale to 0..255 first.
                if np.issubdtype(arr.dtype, np.floating):
                    mx = float(np.max(arr)) if arr.size else 0.0
                    if mx <= 1.5:
                        arr = arr * 255.0
                    arr = np.clip(arr, 0, 255)
                else:
                    arr = np.clip(arr, 0, 255)

                arr = arr.astype(np.uint8, copy=False)

            return np.ascontiguousarray(arr)

        def _to_gray_2d(bgr: np.ndarray) -> np.ndarray:
            """Return pure 2D grayscale (H,W)."""
            if bgr is None or getattr(bgr, "size", 0) == 0:
                return bgr
            if len(bgr.shape) == 2:
                gray = bgr
            elif len(bgr.shape) == 3 and bgr.shape[2] == 3:
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            elif len(bgr.shape) == 3 and bgr.shape[2] == 1:
                gray = bgr[:, :, 0]
            else:
                gray = bgr if len(bgr.shape) == 2 else bgr[:, :, 0]
            return _ensure_uint8_contiguous(gray)

        def _to_gray_hwc1(bgr: np.ndarray) -> np.ndarray:
            """Return grayscale in HWC with 1 channel (H,W,1)."""
            gray2d = _to_gray_2d(bgr)
            if gray2d is None or getattr(gray2d, "size", 0) == 0:
                return gray2d
            if len(gray2d.shape) == 2:
                return _ensure_uint8_contiguous(gray2d[:, :, None])
            return _ensure_uint8_contiguous(gray2d)

        def _to_rgb(bgr: np.ndarray) -> np.ndarray:
            if bgr is None or getattr(bgr, "size", 0) == 0:
                return bgr
            if len(bgr.shape) == 3 and bgr.shape[2] == 3:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            elif len(bgr.shape) == 2:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_GRAY2RGB)
            elif len(bgr.shape) == 3 and bgr.shape[2] == 1:
                rgb = cv2.cvtColor(bgr[:, :, 0], cv2.COLOR_GRAY2RGB)
            else:
                rgb = bgr
            return _ensure_uint8_contiguous(rgb)

        def _looks_like_channel_mismatch(msg: str) -> bool:
            """Detect ONNX input shape/channel mismatch across different ONNXRuntime wordings."""
            m = (msg or "").lower()
            if (
                "invalid dimension" in m
                or "invalid dimensions" in m
                or "shape" in m
                or "dimension" in m
            ):
                if "expected" in m and (
                    "got" in m or "actual" in m or "given" in m or "received" in m
                ):
                    return True
            if "shape mismatch" in m and "expected" in m:
                return True
            if "expected:" in m and "got:" in m:
                return True
            return False

        # Order: most hub EU models expect 1-channel NHWC.
        candidates: list[tuple[str, list[np.ndarray]]] = [
            ("GRAY_HWC1", [_to_gray_hwc1(img) for img in images]),
            ("GRAY2D", [_to_gray_2d(img) for img in images]),
            ("RGB", [_to_rgb(img) for img in images]),
        ]

        last_err: Exception | None = None
        out = None
        for tag, batch in candidates:
            try:
                out = self.recognizer.run(batch)
                last_err = None
                break
            except Exception as e:
                last_err = e
                msg = str(e)
                # If it's *not* a channel/dim mismatch, don't mask the real error.
                if not _looks_like_channel_mismatch(msg):
                    raise
                if DEBUG_LOGS:
                    print(
                        f"[OCR DEBUG] FastPlate input variant '{tag}' failed: {msg}"
                    )
                continue

        if last_err is not None:
            raise last_err

        # Normalize whatever the library returns to `list[str]`.
        if out is None:
            return []
        if isinstance(out, str):
            return [out]
        if isinstance(out, (list, tuple)):
            return [str(x) for x in out]
        if isinstance(out, np.ndarray):
            try:
                tolist_result = out.tolist()
                if isinstance(tolist_result, (list, tuple)):
                    return [str(x) for x in tolist_result]
                if tolist_result is None:
                    return []
                return [str(tolist_result)]
            except Exception:
                return [str(out)]
        return [str(out)]


def _first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        try:
            if p.exists():
                return p
        except Exception:
            continue
    return None


def _find_first_rglob(root: Path, patterns: list[str]) -> Path | None:
    """Find first file under `root` matching any of the glob patterns."""
    for pat in patterns:
        try:
            for p in cast(Any, root).rglob(pat):
                if p.is_file():
                    return p
        except Exception:
            continue
    return None


def _get_fastplate():
    """Load and return the fast-plate-ocr model, or None if unavailable."""
    global _MODEL_CACHE, _MODEL_LOAD_ATTEMPTED

    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    if _MODEL_LOAD_ATTEMPTED:
        return None

    _MODEL_LOAD_ATTEMPTED = True

    base_path = Path(__file__).resolve().parent.parent

    # --- 1) Try custom ONNX export (fastest + best if fine-tuned) ---
    fixed_onnx = base_path / "trained_models/fastplate.onnx"
    fixed_cfg = base_path / "fastplate_assets/plate_pl.yaml"

    found_onnx = _first_existing([fixed_onnx]) or _find_first_rglob(
        base_path / "trained_models",
        patterns=["*.onnx"],
    )

    cfg_candidates: list[Path] = []
    if fixed_cfg.exists():
        cfg_candidates.append(fixed_cfg)
    if found_onnx is not None:
        cfg_candidates.extend(
            [
                found_onnx.parent / "plate_config.yaml",
                found_onnx.parent / "plate_config.yml",
            ]
        )
    found_cfg = _first_existing(cfg_candidates)

    if found_onnx is not None and found_cfg is not None:
        try:
            print(f"[OCR] FastPlate: loading CUSTOM ONNX: {found_onnx}")
            _MODEL_CACHE = _FastPlateOCRWrapper(
                onnx_model_path=str(found_onnx),
                plate_config_path=str(found_cfg),
                device="auto",
            )
            return _MODEL_CACHE
        except ImportError as e:
            print(
                "[OCR ERROR] FastPlate import failed. Install ONNX runtime extras: "
                "pip install 'fast-plate-ocr[onnx]'\n"
                f"Details: {e}"
            )
        except Exception as e:
            print(f"[OCR ERROR] Failed to load custom FastPlate ONNX: {e}")
    else:
        if DEBUG_LOGS:
            print(
                f"[OCR DEBUG] Custom ONNX not found or missing config. onnx={found_onnx} cfg={found_cfg}"
            )

    # --- 2) Fallback: load a model from the official HUB ---
    hub_model = "european-plates-mobile-vit-v2-model"
    try:
        print(f"[OCR] FastPlate: loading HUB model: {hub_model}")
        _MODEL_CACHE = _FastPlateOCRWrapper(hub_model=hub_model, device="auto")
        return _MODEL_CACHE
    except ImportError as e:
        print(
            "[OCR ERROR] FastPlate import failed. Install ONNX runtime extras: "
            "pip install 'fast-plate-ocr[onnx]'\n"
            f"Details: {e}"
        )
    except Exception as e:
        if DEBUG_LOGS:
            print(f"[OCR DEBUG] HUB model load failed: {e}")

    if DEBUG_LOGS:
        print("[OCR WARNING] FastPlate not loaded -> using only Tesseract fallback.")

    return None


def ocr_plate(plate_bgr: np.ndarray) -> str:
    if plate_bgr is None or plate_bgr.size == 0:
        return ""

    if DEBUG_LOGS:
        try:
            mn = float(np.min(plate_bgr))
            mx = float(np.max(plate_bgr))
            print(
                f"[OCR DEBUG] plate dtype={plate_bgr.dtype} min={mn:.3f} max={mx:.3f} shape={plate_bgr.shape}"
            )
        except Exception:
            print(
                f"[OCR DEBUG] plate dtype={getattr(plate_bgr, 'dtype', None)} shape={getattr(plate_bgr, 'shape', None)}"
            )

    # ======================================
    # PRIORYTET: FASTPLATE (AI)
    # ======================================
    model = _get_fastplate()
    ai_clean = ""
    ai_score = 0
    used_engine = "NONE"

    if model:
        try:
            preds = model.recognize_batch([plate_bgr])
            if preds:
                ai_raw = str(preds[0])
                ai_clean = _clean_text(ai_raw)
                ai_score = _score_text(ai_clean)
                used_engine = "FASTPLATE"
        except Exception as e:
            if DEBUG_LOGS:
                print(f"[OCR ERROR] FastPlate exception: {e}")

    # Jeśli AI wygląda dobrze — kończymy od razu (żeby było szybciej)
    if ai_clean and ai_score >= 80:
        if DEBUG_LOGS:
            print(f"[OCR] {used_engine}: {ai_clean} (Score: {ai_score})")
        return ai_clean

    # ======================================
    # FALLBACK: TESSERACT
    # ======================================
    tess_clean = ""
    tess_score = 0
    try:
        import pytesseract

        inp_tess = _preprocess_tess(plate_bgr)
        cfg = r"--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        tess_raw = pytesseract.image_to_string(inp_tess, config=cfg).strip()
        tess_clean = _clean_text(tess_raw)
        tess_score = _score_text(tess_clean)
    except Exception as e:
        if DEBUG_LOGS:
            print(f"[OCR ERROR] Tesseract failed: {e}")

    # ======================================
    # WYBÓR WYNIKU
    # ======================================
    final_text = ""

    if ai_clean and (ai_score >= tess_score):
        final_text = ai_clean
        used_engine = "FASTPLATE (Low Score)"
    elif tess_clean:
        final_text = tess_clean
        used_engine = "TESSERACT"
    elif ai_clean:
        final_text = ai_clean
        used_engine = "FASTPLATE (Fallback)"

    if DEBUG_LOGS:
        print(f"[OCR] {used_engine}: '{final_text}' (AI: {ai_score}, TESS: {tess_score})")

    return final_text


# Alias dla kompatybilności z src/eval.py
ocr_ensemble_plate = ocr_plate


_LPRNET_CACHE = None


def ocr_lprnet_plate(
    plate_bgr: np.ndarray, weights_path: str | None = None, device: str = "auto"
) -> str:
    """Wrapper na LPRNet do użycia w eval.py.
    Wymaga folderu LPRNet_Pytorch w ścieżce (PYTHONPATH).
    """

    global _LPRNET_CACHE

    try:
        import torch
        from LPRNet_Pytorch.model.LPRNet import build_lprnet
        from LPRNet_Pytorch.data.load_data import CHARS
    except ImportError:
        if DEBUG_LOGS:
            print(
                "[OCR ERROR] Brak modułów LPRNet_Pytorch. Upewnij się, że folder LPRNet_Pytorch jest w ścieżce."
            )
        return ""

    if _LPRNET_CACHE is None:
        if not weights_path:
            base_path = Path(__file__).resolve().parent.parent
            weights_path = str(base_path / "LPRNet_Pytorch/weights/Final_LPRNet_model.pth")

        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        lprnet = build_lprnet(
            lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0
        )
        lprnet.to(device)

        if os.path.exists(weights_path):
            try:
                lprnet.load_state_dict(torch.load(weights_path, map_location=device))
                lprnet.eval()
                _LPRNET_CACHE = (lprnet, device)
                if DEBUG_LOGS:
                    print(f"[OCR] Załadowano LPRNet: {weights_path} ({device})")
            except Exception as e:
                print(f"[OCR ERROR] Błąd ładowania wag LPRNet: {e}")
                return ""
        else:
            print(f"[OCR ERROR] Brak pliku wag LPRNet: {weights_path}")
            return ""

    net, dev = _LPRNET_CACHE

    try:
        img = cv2.resize(plate_bgr, (94, 24))
        img = img.astype("float32")
        img -= 127.5
        img *= 0.0078125
        img = np.transpose(img, (2, 0, 1))  # (3, 24, 94)
        img_t = torch.from_numpy(img).unsqueeze(0).to(dev)

        with torch.no_grad():
            preds = net(img_t)
            preds = preds.cpu().detach().numpy()  # (1, T, C)

        labels = np.argmax(preds, axis=2)[0]
        no_repeat_blank = []
        prev_c = -1
        blank = len(CHARS) - 1
        for c in labels:
            if (prev_c == c) or (c == blank):
                prev_c = c
                continue
            no_repeat_blank.append(c)
            prev_c = c

        return "".join([CHARS[c] for c in no_repeat_blank])

    except Exception as e:
        if DEBUG_LOGS:
            print(f"[OCR ERROR] LPRNet inference failed: {e}")
        return ""