# automatic_plate_number_recognition

End-to-end **Automatic License Plate Recognition (ALPR/ANPR)** pipeline in Python:
**images downloaded via script → YOLO license plate detection → FastPlate OCR** for plate text recognition.

## Overview

This repository provides a simple ALPR workflow:
1. **Download images** (cars with visible license plates) using a script
2. **Detect license plates** with **YOLO** (bounding boxes)
3. **Crop plate regions** from the original images
4. **Recognize plate text** using **FastPlate (OCR)**
5. **Return/save results** (predicted license plate string)

## Features

- Dataset download & preparation scripts
- YOLO-based license plate detection
- FastPlate OCR for plate text recognition
- Project structure ready for experimentation (training/evaluation/helpers)

## Project structure

> Names may differ slightly depending on your local setup.

- `src/` — core pipeline code (detection + OCR)
- `scripts/` — helper scripts (download / preprocessing / utilities)
- `data/` — local dataset directory (not always committed)
- `lpr_data/` — prepared data/artifacts for LPR-related experiments
- `fastplate_assets/` — FastPlate assets/models/resources
- `requirements.txt` — Python dependencies
- `download_dataset.py` — dataset/image download entry script
- `prepare_lprnet_data.py` — data preparation helper

## Quick start

### 1) Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt
```


