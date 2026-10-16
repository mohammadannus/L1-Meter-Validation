"""
Proxy module exposing meter detection & cropping from meter_img_detection.
"""

from meter_img_detection import (
    MeterDetector,
    get_meter_detector,
    detect_and_crop_meter,
)

__all__ = ["MeterDetector", "get_meter_detector", "detect_and_crop_meter"]
