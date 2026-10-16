"""
Green Light Detection for MeterSerial-VLM.

Extracts the Display region from a meter image using YOLO, then
uses HSV colour analysis to determine whether the green LED indicator
is ON (connected) or OFF (disconnected).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Project root directory
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Model weights path
_YOLO_MODEL_PATH = str(_PROJECT_ROOT / "L1v1-anas.pt")

# Minimum green-pixel percentage to classify as "connected"
_DEFAULT_GREEN_THRESHOLD = 1.0

# Minimum pixel count — avoids false positives on tiny spurious blobs
_MIN_GREEN_PIXELS = 60


# ---------------------------------------------------------------------------
# Lazy YOLO loader (loaded once, reused across requests)
# ---------------------------------------------------------------------------

_yolo_model: Any | None = None


def _get_yolo_model() -> Any:
    """Return the YOLO model, loading it on first call."""
    global _yolo_model
    if _yolo_model is None:
        try:
            from ultralytics import YOLO  # type: ignore
            _yolo_model = YOLO(_YOLO_MODEL_PATH)
            logger.info(f"✅ Green-light YOLO model loaded: {_YOLO_MODEL_PATH}")
        except Exception as exc:
            logger.error(f"❌ Failed to load YOLO model '{_YOLO_MODEL_PATH}': {exc}")
            raise
    return _yolo_model


# ---------------------------------------------------------------------------
# Core detection functions
# ---------------------------------------------------------------------------

def _detect_green_light(
    display_crop: np.ndarray,
    threshold_percent: float = _DEFAULT_GREEN_THRESHOLD,
) -> tuple[str, float]:
    """
    Analyse a cropped Display region for green-light presence.

    Parameters
    ----------
    display_crop : np.ndarray
        BGR image of the Display region detected by YOLO.
    threshold_percent : float
        Minimum percentage of green pixels required to classify as connected.

    Returns
    -------
    status : str
        ``"powered_on"`` if green light is ON, ``"powered_off"`` otherwise.
    percent : float
        Percentage of pixels that matched the green hue range.
    """
    if display_crop is None or display_crop.size == 0:
        return "powered_off", 0.0

    hsv = cv2.cvtColor(display_crop, cv2.COLOR_BGR2HSV)

    # Green hue range (HSV)
    lower_green = np.array([38, 110, 110])
    upper_green = np.array([82, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Keep only bright pixels (removes dark-green shadows)
    bright_mask = (hsv[:, :, 2] > 140).astype(np.uint8) * 255
    clean_mask = cv2.bitwise_and(mask, bright_mask)

    # Remove small noise blobs
    kernel = np.ones((3, 3), np.uint8)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel)

    green_pixels = cv2.countNonZero(clean_mask)
    total = display_crop.shape[0] * display_crop.shape[1]
    percent = (green_pixels / total * 100) if total > 0 else 0.0

    # Ignore tiny clusters — likely noise
    if green_pixels < _MIN_GREEN_PIXELS:
        percent = 0.0

    status = "powered_on" if percent > threshold_percent else "powered_off"
    return status, round(percent, 2)


def run_green_light_detection(
    image_path: Path,
    threshold_percent: float = _DEFAULT_GREEN_THRESHOLD,
) -> dict[str, Any]:
    """
    Run full green-light detection pipeline on a meter image.

    Steps:
    1. Load image from *image_path*.
    2. Run YOLO to locate the ``Display`` region.
    3. Pass the largest Display crop to :func:`_detect_green_light`.

    Parameters
    ----------
    image_path : Path
        Path to the meter image file.
    threshold_percent : float
        Green-pixel threshold (%) for ON/OFF classification.

    Returns
    -------
    dict with keys:
        ``green_light_status`` — ``"powered_on"`` | ``"powered_off"`` | ``"unknown"``
        ``power_status`` — ``"powered_on"`` | ``"powered_off"`` | ``"unknown"``
        ``green_light_percent`` — float (0.0 if display not found)
    """
    unknown_result: dict[str, Any] = {
        "green_light_status": "unknown",
        "power_status": "unknown",
        "green_light_percent": 0.0,
    }

    # --- Load image --------------------------------------------------------
    try:
        img_bgr = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        img_bgr = None
    if img_bgr is None:
        img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        logger.warning(f"Green-light detection: could not read image '{image_path}'")
        return unknown_result

    # --- YOLO inference ----------------------------------------------------
    try:
        model = _get_yolo_model()
        results = model.predict(img_bgr, conf=0.25, iou=0.45, verbose=False)
    except Exception as exc:
        logger.error(f"Green-light YOLO inference failed: {exc}")
        return unknown_result

    # --- Collect Display crops ---------------------------------------------
    display_crops: list[np.ndarray] = []
    for box in results[0].boxes:
        cls_name = model.names[int(box.cls[0])]
        if cls_name == "Display":
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = img_bgr[y1:y2, x1:x2]
            if crop.size > 0:
                display_crops.append(crop)

    if not display_crops:
        logger.info("Green-light detection: no Display region found in image")
        return unknown_result

    # Use the largest Display region (by area)
    display_crop = max(display_crops, key=lambda c: c.shape[0] * c.shape[1])

    # --- Colour analysis ---------------------------------------------------
    status, percent = _detect_green_light(display_crop, threshold_percent)

    logger.info(
        f"Green-light power detection → status={status}, "
        f"green_percent={percent}%, threshold={threshold_percent}%"
    )

    return {
        "green_light_status": status,
        "power_status": status,
        "green_light_percent": percent,
    }
