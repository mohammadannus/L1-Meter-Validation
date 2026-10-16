import logging
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from models.base.base_model import BaseMeterModel
from models.registry import ModelRegistry
from utils.ocr_extractor import UniversalOCRExtractor
from meter_img_detection import detect_and_crop_meter
from api.ocr_engine import OCREngine

logger = logging.getLogger(__name__)

# Suppress network check overhead on PaddleX models
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

class PaddleOCRBackend(BaseMeterModel):
    """
    Optimized PaddleOCR backend for electricity-meter text recognition.

    Uses YOLO meter detection ('meter_img_detection.pt') to crop unnecessary background
    and runs high-speed PP-OCRv4 on the single cropped meter image.
    """

    def __init__(self) -> None:

        super().__init__("paddleocr")

        self._engine: OCREngine | None = None
        self.is_loaded = False
        self.extractor = UniversalOCRExtractor()

        # Multi-scale tile configuration
        self.tile_rows = 2
        self.tile_columns = 3
        self.tile_overlap = 0.35

    def load(self, model_path: Path | None = None) -> None:
        """
        Initialize using the shared OCREngine singleton.

        The OCREngine loads PaddleOCR once and is shared across
        all API endpoints (meter, invoice, generic OCR).
        """
        self._engine = OCREngine.get_instance()
        self._engine.load()

        self.device = self._engine.device
        self.is_loaded = True

    def unload(self) -> None:
        """
        Mark backend as unloaded.

        Note: The shared OCREngine is NOT unloaded here because
        other endpoints may still be using it.
        """

        self._engine = None
        self.is_loaded = False

    def _create_tiles(
        self,
        image: Image.Image,
    ) -> list[Image.Image]:
        """
        Create overlapping padded tiles.

        Strategy:
            - 2 rows x 3 columns
            - 35% overlap
            - 10% padding around every tile
            - Edge tiles are padded rather than clipped

        The padding is synthetic image space. It does not recover
        pixels outside the original image; overlap is responsible
        for ensuring text near boundaries appears in another tile.
        """

        width, height = image.size

        rows = self.tile_rows
        columns = self.tile_columns

        overlap = self.tile_overlap

        # ---------------------------------------------------------
        # Nominal tile dimensions.
        # ---------------------------------------------------------

        tile_width = math.ceil(
            width / (
                columns
                - overlap * (columns - 1)
            )
        )

        tile_height = math.ceil(
            height / (
                rows
                - overlap * (rows - 1)
            )
        )

        # ---------------------------------------------------------
        # Distance between tile origins.
        # ---------------------------------------------------------

        step_x = int(
            tile_width * (1.0 - overlap)
        )

        step_y = int(
            tile_height * (1.0 - overlap)
        )

        # ---------------------------------------------------------
        # Proportional padding.
        # ---------------------------------------------------------

        padding_x = max(
            16,
            int(tile_width * 0.10),
        )

        padding_y = max(
            16,
            int(tile_height * 0.10),
        )
    

        tiles: list[Image.Image] = []

        for row in range(rows):

            for column in range(columns):

                # -------------------------------------------------
                # Desired crop position.
                # -------------------------------------------------

                x1 = column * step_x
                y1 = row * step_y

                x2 = x1 + tile_width
                y2 = y1 + tile_height

                # -------------------------------------------------
                # Shift the final tile so that it reaches the
                # image boundary instead of leaving uncovered
                # space.
                # -------------------------------------------------

                if column == columns - 1:
                    x2 = width
                    x1 = max(
                        0,
                        x2 - tile_width,
                    )

                if row == rows - 1:
                    y2 = height
                    y1 = max(
                        0,
                        y2 - tile_height,
                    )

                # -------------------------------------------------
                # Actual pixels available inside the image.
                # -------------------------------------------------

                crop_x1 = max(
                    0,
                    x1,
                )

                crop_y1 = max(
                    0,
                    y1,
                )

                crop_x2 = min(
                    width,
                    x2,
                )

                crop_y2 = min(
                    height,
                    y2,
                )

                crop = image.crop(
                    (
                        crop_x1,
                        crop_y1,
                        crop_x2,
                        crop_y2,
                    )
                )

                # -------------------------------------------------
                # Create padded canvas.
                # -------------------------------------------------

                padded_width = (
                    crop.width
                    + padding_x * 2
                )

                padded_height = (
                    crop.height
                    + padding_y * 2
                )

                padded = Image.new(
                    "RGB",
                    (
                        padded_width,
                        padded_height,
                    ),
                    "white",
                )

                padded.paste(
                    crop,
                    (
                        padding_x,
                        padding_y,
                    ),
                )

                tiles.append(
                    padded
                )

        return tiles

    def preprocess(
        self,
        image_path: Path,
        prompt: str | None = None,
    ) -> dict[str, Any]:
        """
        Detect meter region, crop out unnecessary background, and cap resolution if needed.
        """

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        # Detect meter body and crop to eliminate background clutter
        cropped_image, crop_meta = detect_and_crop_meter(image_path)

        # Cap max dimension to 1920px for high-throughput OCR
        max_dim = 1920
        if max(cropped_image.width, cropped_image.height) > max_dim:
            scale = max_dim / max(cropped_image.width, cropped_image.height)
            new_size = (int(cropped_image.width * scale), int(cropped_image.height * scale))
            cropped_image = cropped_image.resize(new_size, Image.Resampling.LANCZOS)

        return {
            "image": cropped_image,
            "crop_meta": crop_meta,
        }

    def _run_ocr(
        self,
        image: Image.Image,
    ) -> str:
        """
        Run PaddleOCR on a single image region.

        Delegates to the shared OCREngine singleton.
        """

        if self._engine is None or not self._engine.is_loaded:
            raise RuntimeError(
                "PaddleOCR model is not loaded."
            )

        return self._engine.run_ocr(image)

    def predict(
        self,
        processed_input: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Run PaddleOCR on the single cropped meter image.
        """

        if not self.is_loaded:
            raise RuntimeError(
                "PaddleOCR model is not loaded."
            )

        image = processed_input[
            "image"
        ]

        crop_meta = processed_input.get(
            "crop_meta",
            {},
        )

        # Run single OCR pass on the cropped meter image
        output = self._run_ocr(
            image
        )

        region_name = (
            "meter_crop"
            if crop_meta.get("meter_detected")
            else "full_image"
        )

        candidates: list[
            dict[str, str]
        ] = [
            {
                "region": region_name,
                "output": output,
            }
        ]

        return {
            "candidates": candidates
        }

    def postprocess(
        self,
        prediction: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Return raw OCR regions.
        """

        regions = []

        for candidate in prediction[
            "candidates"
        ]:

            regions.append(
                {
                    "region":
                        candidate["region"],
                    "raw_output":
                        candidate["output"],
                }
            )

        return {
            "model": self.model_name,
            "regions": regions,
        }

    def get_model_info(
        self,
    ) -> dict[str, Any]:

        return {
            "name": self.model_name,
            "device": str(getattr(self, "device", "cpu")),
            "loaded": self.is_loaded,
            "tile_rows":
                self.tile_rows,
            "tile_columns":
                self.tile_columns,
            "tile_overlap":
                self.tile_overlap,
        }


ModelRegistry.register(
    "paddleocr",
    PaddleOCRBackend,
)