"""
Meter Image Detection & Cropping Module for MeterSerial-VLM.

Identifies the electricity meter region in an image using YOLO ('meter_img_detection.pt')
and crops the meter to eliminate surrounding background, enabling high-speed,
single-image OCR extraction.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration & Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent
_DEFAULT_MODEL_PATH = _PROJECT_ROOT / "meter_img_detection.pt"
_DEFAULT_CONFIDENCE = 0.25
_DEFAULT_PADDING_PCT = 0.02  # 2% padding margin around bounding box to protect edge characters


# ---------------------------------------------------------------------------
# MeterDetector Class
# ---------------------------------------------------------------------------

class MeterDetector:
    """
    YOLO-based Meter Detector that localizes and crops the meter body.
    """

    def __init__(
        self,
        model_path: Path | str | None = None,
        confidence: float = _DEFAULT_CONFIDENCE,
        padding_pct: float = _DEFAULT_PADDING_PCT,
    ) -> None:
        self.model_path = Path(model_path) if model_path else _DEFAULT_MODEL_PATH
        self.confidence = confidence
        self.padding_pct = padding_pct
        self._model: Any | None = None

    def load(self) -> Any:
        """Load the YOLO meter detection model (lazy-loaded)."""
        if self._model is None:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"Meter detection weights not found at: {self.model_path}"
                )
            try:
                from ultralytics import YOLO  # type: ignore

                self._model = YOLO(str(self.model_path))
                logger.info(f"✅ Meter detection YOLO model loaded: {self.model_path}")
            except Exception as exc:
                logger.error(f"❌ Failed to load meter detection model '{self.model_path}': {exc}")
                raise
        return self._model

    def detect(
        self,
        image_input: Image.Image | np.ndarray | Path | str,
        conf: float | None = None,
    ) -> dict[str, Any] | None:
        """
        Detect meter bounding box in an image.

        Returns
        -------
        dict with keys: 'box' [x1, y1, x2, y2], 'confidence', 'class_name'
        or None if no meter detected.
        """
        model = self.load()
        conf_thresh = conf if conf is not None else self.confidence

        # Convert image input to format accepted by YOLO (BGR numpy or path)
        if isinstance(image_input, (str, Path)):
            img_to_predict: Any = str(image_input)
        elif isinstance(image_input, Image.Image):
            img_to_predict = cv2.cvtColor(np.array(image_input.convert("RGB")), cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            img_to_predict = image_input
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        try:
            results = model.predict(img_to_predict, conf=conf_thresh, verbose=False)
        except Exception as exc:
            logger.error(f"Meter detection inference failed: {exc}")
            return None

        if not results or len(results[0].boxes) == 0:
            return None

        boxes = results[0].boxes
        best_box = None
        best_conf = -1.0

        for box in boxes:
            box_conf = float(box.conf[0].item())
            # Find the detection with highest confidence
            if box_conf > best_conf:
                best_conf = box_conf
                cls_id = int(box.cls[0].item())
                cls_name = model.names.get(cls_id, "meter")
                best_box = {
                    "box": [round(coord, 2) for coord in box.xyxy[0].tolist()],
                    "confidence": round(best_conf, 4),
                    "class_name": cls_name,
                }

        return best_box

    def crop_meter(
        self,
        image_input: Image.Image | np.ndarray | Path | str,
        padding_pct: float | None = None,
        conf: float | None = None,
    ) -> tuple[Image.Image, dict[str, Any]]:
        """
        Detect and crop the meter from the image.

        If a meter is detected, crops the meter region with optional padding.
        If no meter is detected (e.g. image already tightly cropped or low confidence),
        falls back safely to the original image.

        Returns
        -------
        (cropped_image: PIL.Image.Image, metadata: dict)
        """
        # Load as PIL Image for consistent output
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            pil_img = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        orig_w, orig_h = pil_img.size
        pad_ratio = padding_pct if padding_pct is not None else self.padding_pct

        detection = self.detect(image_input, conf=conf)

        if detection is None:
            logger.info("ℹ️ No meter bounding box detected. Falling back to full image.")
            return pil_img, {
                "meter_detected": False,
                "confidence": 0.0,
                "raw_box": None,
                "crop_box": [0, 0, orig_w, orig_h],
                "original_size": (orig_w, orig_h),
                "crop_size": (orig_w, orig_h),
            }

        x1, y1, x2, y2 = detection["box"]

        # Apply padding margin
        bw = x2 - x1
        bh = y2 - y1
        pad_x = bw * pad_ratio
        pad_y = bh * pad_ratio

        cx1 = max(0, int(round(x1 - pad_x)))
        cy1 = max(0, int(round(y1 - pad_y)))
        cx2 = min(orig_w, int(round(x2 + pad_x)))
        cy2 = min(orig_h, int(round(y2 + pad_y)))

        cropped = pil_img.crop((cx1, cy1, cx2, cy2))

        logger.info(
            f"🎯 Meter detected with conf={detection['confidence']} at [{x1}, {y1}, {x2}, {y2}]. "
            f"Cropped to [{cx1}, {cy1}, {cx2}, {cy2}] ({cropped.width}x{cropped.height})"
        )

        return cropped, {
            "meter_detected": True,
            "confidence": detection["confidence"],
            "raw_box": [x1, y1, x2, y2],
            "crop_box": [cx1, cy1, cx2, cy2],
            "original_size": (orig_w, orig_h),
            "crop_size": (cropped.width, cropped.height),
        }


# ---------------------------------------------------------------------------
# Global Singleton & Convenience Functions
# ---------------------------------------------------------------------------

_global_meter_detector: MeterDetector | None = None


def get_meter_detector() -> MeterDetector:
    """Return the global MeterDetector singleton instance."""
    global _global_meter_detector
    if _global_meter_detector is None:
        _global_meter_detector = MeterDetector()
    return _global_meter_detector


def detect_and_crop_meter(
    image_input: Image.Image | np.ndarray | Path | str,
    padding_pct: float = _DEFAULT_PADDING_PCT,
    conf: float = _DEFAULT_CONFIDENCE,
) -> tuple[Image.Image, dict[str, Any]]:
    """
    Convenience function to detect and crop the meter from an image.
    Uses the global singleton MeterDetector.
    """
    detector = get_meter_detector()
    return detector.crop_meter(image_input, padding_pct=padding_pct, conf=conf)


# ---------------------------------------------------------------------------
# CLI Usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Detect and crop meter region from image.")
    parser.add_argument("image_path", type=str, help="Path to input meter image")
    parser.add_argument("--output", "-o", type=str, default=None, help="Path to save cropped image")
    parser.add_argument("--conf", "-c", type=float, default=_DEFAULT_CONFIDENCE, help="Confidence threshold")
    parser.add_argument("--pad", "-p", type=float, default=_DEFAULT_PADDING_PCT, help="Padding ratio (e.g. 0.02)")

    args = parser.parse_args()

    input_path = Path(args.image_path)
    if not input_path.exists():
        print(f"Error: Image '{input_path}' not found.", file=sys.stderr)
        sys.exit(1)

    crop_img, meta = detect_and_crop_meter(input_path, padding_pct=args.pad, conf=args.conf)

    print("Meter Detection Result:")
    print(f"  Detected     : {meta['meter_detected']}")
    print(f"  Confidence   : {meta['confidence']}")
    print(f"  Raw Box      : {meta['raw_box']}")
    print(f"  Crop Box     : {meta['crop_box']}")
    print(f"  Original Size: {meta['original_size']}")
    print(f"  Crop Size    : {meta['crop_size']}")

    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        crop_img.save(out_p)
        print(f"  Saved crop to: {out_p}")
