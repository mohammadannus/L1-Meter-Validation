"""
Model inference orchestration for the MeterSerial-VLM API.

Manages a single loaded backend at a time to conserve GPU memory.
Backends are loaded lazily on first request and persist across calls.
"""

from __future__ import annotations

import importlib
import logging
import threading
import time
from pathlib import Path
from typing import Any

from models.registry import ModelRegistry
from utils.ocr_extractor import UniversalOCRExtractor
from utils.ocr_consolidator import UniversalOCRConsolidator
from utils.green_light import run_green_light_detection
from utils.seal_detection import run_seal_detection

logger = logging.getLogger(__name__)

# Map backend model names to module paths
_BACKEND_MODULES: dict[str, str] = {
    "paddleocr": "models.backends.paddleocr.paddleocr_backend",
}

def _register_backend_module(model_name: str) -> None:
    """Import backend module to register it with ModelRegistry if not already registered."""
    if ModelRegistry.is_registered(model_name):
        return
    module_path = _BACKEND_MODULES.get(model_name)
    if module_path:
        try:
            importlib.import_module(module_path)
            logger.info(f"Registered backend '{model_name}' from {module_path}")
        except Exception as err:
            logger.warning(f"Could not load backend '{model_name}': {err}")

# Register PaddleOCR backend
_register_backend_module("paddleocr")

# ---------------------------------------------------------------------------
# Configuration & Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Model priority order (first is default)
AVAILABLE_MODELS: list[str] = ["paddleocr"]
DEFAULT_MODEL: str = "paddleocr"

# Where locally downloaded models live
MODEL_STORE_ROOT = _PROJECT_ROOT / "model_store"


# ---------------------------------------------------------------------------
# Singleton state
# ---------------------------------------------------------------------------

_backend_lock = threading.Lock()
_current_model_name: str | None = None
_current_backend: Any = None
_extractor = UniversalOCRExtractor()
_consolidator = UniversalOCRConsolidator()


def get_available_models() -> list[str]:
    """Return list of configured model names."""
    return AVAILABLE_MODELS


def get_default_model() -> str:
    """Return the default model name."""
    return DEFAULT_MODEL


def get_current_model_name() -> str | None:
    """Return the name of the currently loaded model backend."""
    return _current_model_name


def _load_backend(model_name: str = "paddleocr") -> Any:
    """
    Load (or reuse) the PaddleOCR backend.
    Thread-safe via _backend_lock.
    """
    global _current_model_name, _current_backend

    with _backend_lock:
        if _current_model_name == model_name and _current_backend is not None:
            return _current_backend

        # Unload previous backend
        if _current_backend is not None:
            try:
                _current_backend.unload()
            except Exception as exc:
                logger.warning(f"Error unloading previous backend: {exc}")
            _current_backend = None
            _current_model_name = None

        # Ensure backend is registered
        _register_backend_module(model_name)

        # Instantiate new backend
        backend_class = ModelRegistry.get_backend(model_name)
        backend = backend_class()
        backend.load()

        _current_backend = backend
        _current_model_name = model_name

        return backend


def run_inference(
    image_path: Path,
    model_name: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """
    Run end-to-end OCR inference on an image.

    Parameters
    ----------
    image_path : Path
        Path to the image file (temporary file from upload).
    model_name : str
        Backend to use. Defaults to DEFAULT_MODEL.

    Returns
    -------
    dict
        Keys: serial_number, imei, model_used, processing_time_ms, regions
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: {', '.join(AVAILABLE_MODELS)}"
        )

    start_time = time.perf_counter()

    backend = _load_backend(model_name)

    # Preprocess
    processed_input = backend.preprocess(image_path)

    # Predict
    prediction = backend.predict(processed_input)

    candidates = prediction.get("candidates", [])

    # Per-region extraction
    extracted_results: dict[str, dict[str, Any]] = {}

    region_details = []

    for candidate in candidates:
        region_name = candidate.get("region", "unknown")
        raw_output = candidate.get("output", "")

        region_extraction = _extractor.extract(raw_output)
        extracted_results[region_name] = region_extraction

        region_details.append(
            {
                "region": region_name,
                "raw_output": raw_output,
                "serial_number": region_extraction.get("serial_number", ""),
                "imei": region_extraction.get("imei", ""),
            }
        )

    # Consolidate
    final = _consolidator.consolidate(extracted_results)

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # ------------------------------------------------------------------
    # Green light / Power status detection (runs on the same image)
    # ------------------------------------------------------------------
    green_result = {"green_light_status": "unknown", "power_status": "unknown", "green_light_percent": 0.0}
    try:
        green_result = run_green_light_detection(image_path)
    except Exception as exc:
        logger.warning(f"Green-light detection skipped due to error: {exc}")

    # ------------------------------------------------------------------
    # Seal detection (runs on the same image, independently)
    # ------------------------------------------------------------------
    seal_result = {
        "seal_count": 0,
        "meter_locked": False,
        "seal_confidences": [],
        "average_seal_confidence": 0.0,
        "seals": [],
    }
    try:
        seal_result = run_seal_detection(image_path)
    except Exception as exc:
        logger.warning(f"Seal detection skipped due to error: {exc}")

    return {
        "serial_number": final.get("serial_number", ""),
        "imei": final.get("imei", ""),
        "green_light_status": green_result.get("green_light_status", "unknown"),
        "green_light_percent": green_result.get("green_light_percent", 0.0),
        "power_status": green_result.get("power_status", green_result.get("green_light_status", "unknown")),
        "seal_count": seal_result.get("seal_count", 0),
        "meter_locked": seal_result.get("meter_locked", False),
        "seal_confidences": seal_result.get("seal_confidences", []),
        "average_seal_confidence": seal_result.get("average_seal_confidence", 0.0),
        "seals": seal_result.get("seals", []),
        "model_used": model_name,
        "processing_time_ms": round(elapsed_ms, 2),
        "regions": region_details,
    }
