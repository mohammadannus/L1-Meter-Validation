# importing libraries

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from PIL import Image


@dataclass
class OCRRegion:
    """
    Represents one image region for OCR.
    """

    name: str
    image: Image.Image
    scale: str


class MultiScaleOCR:
    """
    Generate multi-scale image regions for OCR.

    Strategy:
    1. Original full image
    2. Upscaled full image
    3. Overlapping high-resolution windows
    """

    def __init__(
        self,
        upscale_factor: float = 2.0,
        window_size: int = 700,
        overlap: float = 0.30,
    ) -> None:

        self.upscale_factor = upscale_factor
        self.window_size = window_size
        self.overlap = overlap

    def generate(
        self,
        image: Image.Image,
    ) -> Iterator[OCRRegion]:
        """
        Generate OCR regions progressively.
        """

        image = image.convert("RGB")

        # ----------------------------------------------------------
        # Scale 1: original full image
        # ----------------------------------------------------------

        yield OCRRegion(
            name="full_original",
            image=image,
            scale="original",
        )

        # ----------------------------------------------------------
        # Scale 2: upscaled full image
        # ----------------------------------------------------------

        upscaled = self._upscale(
            image
        )

        yield OCRRegion(
            name="full_upscaled",
            image=upscaled,
            scale="upscaled",
        )

        # ----------------------------------------------------------
        # Scale 3: overlapping high-resolution windows
        # ----------------------------------------------------------

        for index, window in enumerate(
            self._create_windows(
                upscaled
            ),
            start=1,
        ):

            yield OCRRegion(
                name=f"window_{index}",
                image=window,
                scale="window",
            )

    def _upscale(
        self,
        image: Image.Image,
    ) -> Image.Image:
        """
        Upscale an image using high-quality Lanczos resampling.
        """

        width, height = image.size

        new_width = int(
            width * self.upscale_factor
        )

        new_height = int(
            height * self.upscale_factor
        )

        return image.resize(
            (
                new_width,
                new_height,
            ),
            Image.Resampling.LANCZOS,
        )

    def _create_windows(
        self,
        image: Image.Image,
    ) -> list[Image.Image]:
        """
        Create overlapping square windows.
        """

        width, height = image.size

        window_size = min(
            self.window_size,
            width,
            height,
        )

        step = int(
            window_size
            * (1.0 - self.overlap)
        )

        if step <= 0:
            raise ValueError(
                "Overlap must be less than 1.0."
            )

        x_positions = self._positions(
            width,
            window_size,
            step,
        )

        y_positions = self._positions(
            height,
            window_size,
            step,
        )

        windows: list[Image.Image] = []

        for y in y_positions:

            for x in x_positions:

                windows.append(
                    image.crop(
                        (
                            x,
                            y,
                            x + window_size,
                            y + window_size,
                        )
                    )
                )

        return windows

    def _positions(
        self,
        total_size: int,
        window_size: int,
        step: int,
    ) -> list[int]:
        """
        Calculate window positions while ensuring
        the final part of the image is covered.
        """

        if total_size <= window_size:
            return [0]

        positions: list[int] = []

        position = 0

        while position + window_size < total_size:

            positions.append(
                position
            )

            position += step

        final_position = (
            total_size - window_size
        )

        if not positions or positions[-1] != final_position:

            positions.append(
                final_position
            )

        return positions