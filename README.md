# MeterSerial-VLM

**L1 Validation system for extracting Serial Numbers and IMEI numbers from electricity meter images.**

Uses multi-scale OCR and Vision-Language Models (VLMs) with a tiling + consolidation strategy to achieve reliable extraction from real-world meter photographs.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Running the Application](#running-the-application)
- [API Reference](#api-reference)
- [CLI Usage](#cli-usage)
- [Environment Variables](#environment-variables)
- [Model Details](#model-details)

---

## Features

- **YOLO Meter Detection & Auto-Cropping** — localizes the meter body (`meter_img_detection.pt`) and eliminates background noise
- **Single-Image High-Speed OCR** — single-pass PP-OCRv4 inference on cropped meter image instead of multi-tile iterations
- **YOLO Meter Seal Detection** — detects physical meter seals (`seal_detection.pt`), determines lock status (`meter_locked`), seal count, and confidence scores
- **Green light power status detection** — YOLO-based display extraction & HSV status classification to identify if the meter is powered ON or OFF
- **Deterministic extraction** — regex-based serial/IMEI extractor supporting diverse label variations (Sr. No., IMEI NO, S.No, etc.)
- **Result consolidation** — candidate verification and validation
- **FastAPI REST API** — JSON response with region breakdown, power status, and seal details
- **Premium frontend UI** — drag-drop upload, live results, copy-to-clipboard, power status indicator & seal lock badges

---

## Architecture

```
Image Input
    │
    ├── Meter Detection & Auto-Cropping (meter_img_detection.pt)
    │       └── Identifies meter bounding box & crops away background clutter
    │       └── (Graceful fallback to full image if no bounding box detected)
    │
    ├── PaddleOCR Inference (Single Cropped Meter Image)
    │       └── PP-OCRv4 text detection & recognition in a single pass
    │       └── UniversalOCRExtractor (regex patterns for Serial / IMEI)
    │       └── UniversalOCRConsolidator (candidate validation → final result)
    │
    ├── Seal Detection (seal_detection.pt)
    │       └── Identifies physical seals on meter body
    │       └── Computes seal count, individual confidences, and locked status
    │
    └── Green Light Power Detection (L1v1-anas.pt)
            └── Locates Display LED region via YOLO
            └── Classifies meter power status (powered_on / powered_off)
```

---

## Project Structure

```
MeterSerial-VLM/
├── api/                        # FastAPI application
│   ├── app.py                  # App factory, routing, static files
│   ├── inference.py            # Model loading + inference orchestration
│   ├── schemas.py              # Pydantic request/response models
│   └── routes/
│       ├── health.py           # GET /api/v1/health, /api/v1/models
│       └── extract.py          # POST /api/v1/extract
│
├── frontend/                   # Browser UI (served by FastAPI)
│   ├── index.html
│   ├── style.css
│   └── app.js
│
├── models/
│   ├── base/base_model.py      # Abstract base class
│   ├── registry.py             # Model registry
│   ├── factory.py              # Model factory
│   └── backends/
│       ├── paddleocr/          # PaddleOCR backend
│       ├── lightonocr/         # LightOnOCR-2-1B backend
│       └── qwen/               # Qwen2-VL-2B backend
│
├── utils/
│   ├── ocr_extractor.py        # UniversalOCRExtractor
│   ├── ocr_consolidator.py     # UniversalOCRConsolidator
│   ├── multiscale_ocr.py       # MultiScaleOCR tiling utility
│   └── model_manager.py        # Model download + lifecycle
│
├── scripts/                    # CLI test scripts
│   ├── test_paddleocr.py
│   ├── test_lightonocr.py
│   └── test_qwen_inference.py  (if applicable)
│
├── configs/
│   └── models.yaml             # Model repo IDs and local paths
│
├── prompts/
│   └── meter_serial_prompt.txt # VLM extraction prompt
│
├── model_store/                # Downloaded model weights (git-ignored)
├── requirements-full.txt
├── .env.example
└── main.py                     # Entry point
```

---

## Setup

### 1. Clone and create virtual environment

```bash
git clone <repo-url>
cd MeterSerial-VLM

python -m venv .venv
```

**Activate:**

```bash
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements-full.txt
```

> **GPU note:** `torch` and `torchvision` in `requirements-full.txt` target CUDA 13.2.
> For a different CUDA version, install torch separately from https://pytorch.org/get-started/locally/

### 3. Install PaddleOCR (optional — separate venv recommended)

PaddleOCR requires PaddlePaddle which may conflict with PyTorch.

```bash
# Create a dedicated venv for PaddleOCR
python -m venv .paddlevenv
.\.paddlevenv\Scripts\Activate.ps1   # Windows

pip install paddlepaddle-gpu paddleocr
```

> To use PaddleOCR through the API, activate the paddle venv instead of `.venv`.

### 4. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and add your Hugging Face token if needed:

```
HF_TOKEN=hf_...
HF_HOME=./model_store/cache
```

### 5. Download model weights

```bash
# Download all models listed in configs/models.yaml
python -c "
from utils.model_manager import ModelManager
mm = ModelManager()
mm.download_all()
"
```

Or download a specific model:

```bash
python -c "
from utils.model_manager import ModelManager
mm = ModelManager()
mm.download('lightonocr')
"
```

---

## Running the Application

### Start the API server + frontend

```bash
python main.py
```

The server starts at **http://localhost:8000**

| URL | Description |
|-----|-------------|
| `http://localhost:8000` | Frontend UI |
| `http://localhost:8000/docs` | Interactive API docs (Swagger) |
| `http://localhost:8000/redoc` | Alternative API docs |

### Using uvicorn directly

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

### Development mode (auto-reload)

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

---

## API Reference

### `GET /api/v1/health`

Health check.

```bash
curl http://localhost:8000/api/v1/health
```

```json
{ "status": "ok", "version": "1.0.0" }
```

---

### `GET /api/v1/models`

List available model backends.

```bash
curl http://localhost:8000/api/v1/models
```

```json
{
  "models": ["paddleocr", "lightonocr", "qwen2_vl"],
  "default_model": "paddleocr"
}
```

---

### `POST /api/v1/extract`

Extract Serial Number and IMEI from a meter image.

**Request** — `multipart/form-data`:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | File | ✅ | Meter image (JPEG, PNG, WEBP, BMP, TIFF, max 20 MB) |
| `model` | string | ❌ | Model backend (`paddleocr` / `lightonocr` / `qwen2_vl`). Default: `paddleocr` |

**Example — curl:**

```bash
curl -X POST http://localhost:8000/api/v1/extract \
  -F "file=@/path/to/meter.jpg" \
  -F "model=paddleocr"
```

**Example — curl with LightOnOCR:**

```bash
curl -X POST http://localhost:8000/api/v1/extract \
  -F "file=@/path/to/meter.jpg" \
  -F "model=lightonocr"
```

**Example — Python requests:**

```python
import requests

with open("meter.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/v1/extract",
        files={"file": f},
        data={"model": "paddleocr"},
    )

result = response.json()
print("Serial Number:", result["serial_number"])
print("IMEI:         ", result["imei"])
```

**Response:**

```json
{
  "serial_number": "U5774255",
  "imei": "864524079619460",
  "green_light_status": "powered_on",
  "green_light_percent": 10.29,
  "power_status": "powered_on",
  "seal_count": 3,
  "meter_locked": true,
  "seal_confidences": [0.8351, 0.8166, 0.7176],
  "average_seal_confidence": 0.7898,
  "seals": [
    {
      "confidence": 0.8351,
      "box": [464.66, 693.75, 548.83, 791.33],
      "class_name": "seal"
    },
    {
      "confidence": 0.8166,
      "box": [737.64, 438.09, 810.59, 514.72],
      "class_name": "seal"
    },
    {
      "confidence": 0.7176,
      "box": [161.80, 436.97, 239.39, 528.20],
      "class_name": "seal"
    }
  ],
  "model_used": "paddleocr",
  "processing_time_ms": 3241.50,
  "regions": [
    {
      "region": "meter_crop",
      "raw_output": "...",
      "serial_number": "U5774255",
      "imei": "864524079619460"
    }
  ]
}
```

**Error responses:**

| Code | Reason |
|------|--------|
| `400` | Unsupported file type / file too large / unknown model |
| `500` | Inference failure |

---

## CLI Usage

Run inference from the command line without starting the server.

### Seal Detection

```bash
python -m utils.seal_detection path/to/meter.jpg
```

### PaddleOCR

```bash
# Activate PaddleOCR venv first
.\.paddlevenv\Scripts\Activate.ps1

python -m scripts.test_paddleocr path/to/meter.jpg
```

### LightOnOCR

```bash
# Activate main venv
.\.venv\Scripts\Activate.ps1

python -m scripts.test_lightonocr path/to/meter.jpg
```

### Qwen2-VL

```bash
python -m scripts.test_qwen_inference path/to/meter.jpg
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | *(empty)* | Hugging Face access token for gated models |
| `HF_HOME` | *(HF default)* | Custom cache directory for HF downloads |

---

## Model Details

| Model | Size | Backend | GPU Required | HF Repo |
|-------|------|---------|--------------|---------|
| PaddleOCR | ~50 MB | `paddleocr` | No (CPU OK) | Auto-downloaded |
| LightOnOCR-2-1B | ~2 GB | `lightonocr` | Recommended | `lightonai/LightOnOCR-2-1B` |
| Qwen2-VL-2B | ~4 GB | `qwen2_vl` | Recommended | `Qwen/Qwen2-VL-2B-Instruct` |

All VLM models are loaded from `model_store/` (set `local_dir` in `configs/models.yaml`).

---

## Extraction Logic

The `UniversalOCRExtractor` identifies Serial Numbers and IMEIs by matching against label patterns:

**Serial Number labels:** `Serial Number`, `Serial No`, `SL NO`, `S/N`, `Meter No`, `Meter ID`, `NO`

**IMEI labels:** `IMEI`, `IMEI NO`, `IMEI Number`

Anti-confusion rules prevent common misidentifications (meter readings, dates, voltages, model numbers).

The `UniversalOCRConsolidator` then merges candidates from all 7 regions using:
- Exact frequency voting
- Prefix/suffix clustering (handles OCR truncation)
- Levenshtein distance tolerance (~10%)
............................................

# 1. Setup
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-full.txt

# 2. Download models
python -c "from utils.model_manager import ModelManager; ModelManager().download_all()"

# 3. Start server + frontend
python main.py
# → http://localhost:8000  (UI)
# → http://localhost:8000/docs  (Swagger API docs)

# 4. CLI tests
python -m scripts.test_paddleocr   data/images/meter.jpg
python -m scripts.test_lightonocr  data/images/meter.jpg
