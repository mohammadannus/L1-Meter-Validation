"""
Shared OCR Engine — Single PaddleOCR instance for all application use cases.

This module provides a thread-safe singleton OCR engine that is loaded once
during application startup and shared across all API endpoints (meter, invoice,
generic OCR). This avoids loading multiple PaddleOCR instances (~500MB+ each).
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    import torch  # noqa: F401
except ImportError:
    pass

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class OCREngine:
    """
    Singleton shared PaddleOCR engine.

    Provides a simple interface:
        engine = OCREngine.get_instance()
        engine.load()
        raw_text = engine.run_ocr(pil_image)
    """

    _instance: OCREngine | None = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._ocr = None
        self._model_lock = threading.Lock()
        self.is_loaded = False
        self.device = "cpu"

    # -----------------------------------------------------------------
    # Singleton access
    # -----------------------------------------------------------------

    @classmethod
    def get_instance(cls) -> OCREngine:
        """Return the singleton OCR engine instance (thread-safe)."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # -----------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------

    def load(self) -> None:
        """Load PaddleOCR model into memory (idempotent)."""
        if self.is_loaded:
            return

        with self._model_lock:
            if self.is_loaded:
                return

            # Suppress network check overhead on PaddleX models
            os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

            has_cuda = False
            try:
                import paddle
                has_cuda = (
                    paddle.device.is_compiled_with_cuda()
                    and paddle.device.cuda.device_count() > 0
                )
            except Exception:
                pass

            device = "gpu" if has_cuda else "cpu"
            ocr_version = os.environ.get("PADDLE_OCR_VERSION", "PP-OCRv4")

            from paddleocr import PaddleOCR

            try:
                # PaddleOCR v3.x / PaddleX format
                self._ocr = PaddleOCR(
                    lang="en",
                    ocr_version=ocr_version,
                    device=device,
                    enable_mkldnn=False,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    text_det_thresh=0.2,
                    text_det_box_thresh=0.4,
                )
            except (TypeError, ValueError) as init_err:
                logger.warning(
                    "Standard v3 PaddleOCR kwargs failed (%s), falling back to v2 kwargs...",
                    init_err,
                )
                # Standard PaddleOCR v2.x format
                self._ocr = PaddleOCR(
                    use_angle_cls=False,
                    lang="en",
                    ocr_version=ocr_version,
                    use_gpu=has_cuda,
                    det_db_thresh=0.2,
                    det_db_box_thresh=0.4,
                )


            self.device = device
            self.is_loaded = True
            logger.info(
                "OCREngine loaded PaddleOCR (%s) on %s",
                ocr_version,
                device,
            )

    def unload(self) -> None:
        """Release PaddleOCR resources."""
        with self._model_lock:
            self._ocr = None
            self.is_loaded = False
            logger.info("OCREngine unloaded.")

    # -----------------------------------------------------------------
    # Core OCR — returns raw text
    # -----------------------------------------------------------------

    def run_ocr(
        self,
        image: Image.Image,
        min_confidence: float = 0.30,
    ) -> str:
        """
        Run PaddleOCR on a PIL image and return raw text.

        Parameters
        ----------
        image : PIL.Image.Image
            Input image (will be converted to RGB).
        min_confidence : float
            Minimum OCR confidence threshold (0.0–1.0).

        Returns
        -------
        str
            Newline-separated OCR text lines.
        """
        if not self.is_loaded or self._ocr is None:
            self.load()

        # PIL RGB → NumPy RGB
        image_array = np.asarray(image.convert("RGB"))

        lines: list[str] = []

        with self._model_lock:
            # 1. PaddleOCR v3.x / PaddleX format (.predict)
            if hasattr(self._ocr, "predict"):
                result = self._ocr.predict(input=image_array)
                for page in result:
                    if page is None:
                        continue
                    data = page.json
                    if callable(data):
                        data = data()
                    if not isinstance(data, dict):
                        continue
                    res = data.get("res", data)
                    if not isinstance(res, dict):
                        continue
                    texts = res.get("rec_texts", [])
                    scores = res.get("rec_scores", [])
                    for index, text in enumerate(texts):
                        if not text:
                            continue
                        text = str(text).strip()
                        if not text:
                            continue
                        if scores:
                            try:
                                score = float(scores[index])
                                if score < min_confidence:
                                    continue
                            except (ValueError, TypeError, IndexError):
                                pass
                        lines.append(text)

            # 2. Standard PaddleOCR v2.x format (.ocr)
            elif hasattr(self._ocr, "ocr"):
                result = self._ocr.ocr(image_array, cls=False)
                # result structure: [[ [ [box], (text, score) ], ... ]]
                if result:
                    for page in result:
                        if not page:
                            continue
                        for line_item in page:
                            if not line_item or len(line_item) < 2:
                                continue
                            text_score = line_item[1]
                            if isinstance(text_score, (list, tuple)) and len(text_score) >= 2:
                                text = str(text_score[0]).strip()
                                try:
                                    score = float(text_score[1])
                                    if score < min_confidence:
                                        continue
                                except (ValueError, TypeError):
                                    pass
                                if text:
                                    lines.append(text)
                            elif isinstance(text_score, str) and text_score.strip():
                                lines.append(text_score.strip())
            else:
                raise AttributeError("Unsupported PaddleOCR object: neither 'predict' nor 'ocr' method found.")


        return "\n".join(lines)

    def run_ocr_from_path(
        self,
        image_path: Path,
        max_dimension: int = 1920,
        min_confidence: float = 0.30,
    ) -> str:
        """
        Open an image from disk, optionally resize, run OCR.

        Parameters
        ----------
        image_path : Path
            Path to the image file.
        max_dimension : int
            Cap the largest dimension to this value.
        min_confidence : float
            Minimum OCR confidence threshold.

        Returns
        -------
        str
            Newline-separated OCR text lines.
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")

        # Cap resolution to prevent OOM on very large images
        if max(image.width, image.height) > max_dimension:
            scale = max_dimension / max(image.width, image.height)
            new_size = (
                int(image.width * scale),
                int(image.height * scale),
            )
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        try:
            return self.run_ocr(image, min_confidence=min_confidence)
        finally:
            image.close()
