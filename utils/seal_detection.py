"""
Meter Seal Detection Module for MeterSerial-VLM.

Identifies physical meter seals in an image using YOLO ('seal_detection.pt').
Multiple seals indicate that the electricity meter is locked/tamper-secured.
Extracts the total seal count, individual confidence scores, and bounding boxes.
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

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_MODEL_PATH = _PROJECT_ROOT / "seal_detection.pt"
_DEFAULT_CONFIDENCE = 0.25


# ---------------------------------------------------------------------------
# SealDetector Class
# ---------------------------------------------------------------------------

class SealDetector:
    """
    YOLO-based Seal Detector that identifies tamper/security seals on meters.
    """

    def __init__(
        self,
        model_path: Path | str | None = None,
        confidence: float = _DEFAULT_CONFIDENCE,
    ) -> None:
        self.model_path = Path(model_path) if model_path else _DEFAULT_MODEL_PATH
        self.confidence = confidence
        self._model: Any | None = None

    def load(self) -> Any:
        """Load the YOLO seal detection model (lazy-loaded singleton)."""
        if self._model is None:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"Seal detection model weights not found at: {self.model_path}"
                )
            try:
                from ultralytics import YOLO  # type: ignore

                self._model = YOLO(str(self.model_path))
                logger.info(f"✅ Seal detection YOLO model loaded: {self.model_path}")
            except Exception as exc:
                logger.error(f"❌ Failed to load seal detection model '{self.model_path}': {exc}")
                raise
        return self._model

    def detect(
        self,
        image_input: Image.Image | np.ndarray | Path | str,
        conf: float | None = None,
    ) -> dict[str, Any]:
        """
        Detect meter seals in an image.

        Returns
        -------
        dict with keys:
            - seal_count: int (number of detected seals)
            - meter_locked: bool (True if seal_count > 0, False otherwise)
            - seal_confidences: list[float] (confidence for each seal)
            - average_seal_confidence: float (mean confidence across detected seals)
            - seals: list[dict] (per-seal details: confidence, box [x1, y1, x2, y2], class_name)
        """
        conf_thresh = conf if conf is not None else self.confidence
        default_result: dict[str, Any] = {
            "seal_count": 0,
            "meter_locked": False,
            "seal_confidences": [],
            "average_seal_confidence": 0.0,
            "seals": [],
        }

        # Convert image input to format accepted by YOLO (path, PIL Image, or BGR numpy)
        if isinstance(image_input, (str, Path)):
            img_path = Path(image_input)
            if not img_path.exists():
                logger.warning(f"Seal detection: image file not found '{img_path}'")
                return default_result
            img_to_predict: Any = str(img_path)
        elif isinstance(image_input, Image.Image):
            img_to_predict = cv2.cvtColor(np.array(image_input.convert("RGB")), cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            img_to_predict = image_input
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        try:
            model = self.load()
            results = model.predict(img_to_predict, conf=conf_thresh, verbose=False)
        except Exception as exc:
            logger.error(f"Seal detection inference failed: {exc}")
            return default_result

        if not results or len(results[0].boxes) == 0:
            logger.info("ℹ️ No meter seals detected.")
            return default_result

        boxes = results[0].boxes
        seals: list[dict[str, Any]] = []
        confidences: list[float] = []

        for box in boxes:
            box_conf = round(float(box.conf[0].item()), 4)
            cls_id = int(box.cls[0].item())
            cls_name = model.names.get(cls_id, "seal")
            coords = [round(float(c), 2) for c in box.xyxy[0].tolist()]

            confidences.append(box_conf)
            seals.append(
                {
                    "confidence": box_conf,
                    "box": coords,
                    "class_name": cls_name,
                }
            )

        # Sort seals by confidence descending
        seals.sort(key=lambda s: s["confidence"], reverse=True)
        confidences.sort(reverse=True)

        seal_count = len(seals)
        avg_conf = round(sum(confidences) / seal_count, 4) if seal_count > 0 else 0.0

        logger.info(
            f"🔒 Seal detection: {seal_count} seal(s) found "
            f"(confidences: {confidences}, avg: {avg_conf}). Meter locked: True"
        )

        return {
            "seal_count": seal_count,
            "meter_locked": seal_count > 0,
            "seal_confidences": confidences,
            "average_seal_confidence": avg_conf,
            "seals": seals,
        }


# ---------------------------------------------------------------------------
# Global Singleton & Convenience Functions
# ---------------------------------------------------------------------------

_global_seal_detector: SealDetector | None = None


def get_seal_detector() -> SealDetector:
    """Return the global SealDetector singleton instance."""
    global _global_seal_detector
    if _global_seal_detector is None:
        _global_seal_detector = SealDetector()
    return _global_seal_detector


def run_seal_detection(
    image_input: Image.Image | np.ndarray | Path | str,
    conf: float = _DEFAULT_CONFIDENCE,
) -> dict[str, Any]:
    """
    Run seal detection pipeline on a meter image.
    Uses the global singleton SealDetector.
    """
    detector = get_seal_detector()
    return detector.detect(image_input, conf=conf)


# ---------------------------------------------------------------------------
# CLI Usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(description="Detect meter seals using YOLO model.")
    parser.add_argument("image_path", type=str, help="Path to input meter image")
    parser.add_argument("--conf", "-c", type=float, default=_DEFAULT_CONFIDENCE, help="Confidence threshold")

    args = parser.parse_args()

    input_path = Path(args.image_path)
    if not input_path.exists():
        print(f"Error: Image '{input_path}' not found.", file=sys.stderr)
        sys.exit(1)

    result = run_seal_detection(input_path, conf=args.conf)

    print("=" * 60)
    print("  METER SEAL DETECTION RESULT")
    print("=" * 60)
    print(f"  Seal Count    : {result['seal_count']}")
    print(f"  Meter Locked  : {'LOCKED' if result['meter_locked'] else 'UNLOCKED'}")
    print(f"  Confidences   : {result['seal_confidences']}")
    print(f"  Avg Confidence: {result['average_seal_confidence']}")
    print("-" * 60)
    print("  Full Output:")
    print(json.dumps(result, indent=2))
    print("=" * 60)
