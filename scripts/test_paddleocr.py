from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from models.backends.paddleocr.paddleocr_backend import (
    PaddleOCRBackend,
)

from utils.ocr_extractor import (
    UniversalOCRExtractor,
)

from utils.ocr_consolidator import (
    UniversalOCRConsolidator,
)


def print_separator(
    character: str = "=",
    width: int = 70,
) -> None:

    print(character * width)


def print_region_header(
    region_name: str,
) -> None:

    print()
    print(
        f"{'-' * 26} "
        f"{region_name} "
        f"{'-' * 26}"
    )
    print()


def print_raw_outputs(
    raw_outputs: dict[str, str],
) -> None:

    print()
    print_separator()

    print(
        "                    PADDLEOCR RAW OUTPUT"
    )

    print_separator()

    for region_name, raw_output in (
        raw_outputs.items()
    ):

        print_region_header(
            region_name
        )

        if raw_output:
            print(
                raw_output.strip()
            )
        else:
            print("NOT_FOUND")

    print()


def print_extraction_results(
    extracted_results: dict[
        str,
        dict[str, Any],
    ],
) -> None:

    print_separator()

    print(
        "                 PYTHON EXTRACTION RESULTS"
    )

    print_separator()

    for region_name, result in (
        extracted_results.items()
    ):

        print_region_header(
            region_name
        )

        serial_number = result.get(
            "serial_number",
            "",
        )

        imei = result.get(
            "imei",
            "",
        )

        print(
            "Serial Number : "
            f"{serial_number or 'NOT_FOUND'}"
        )

        print(
            "IMEI          : "
            f"{imei or 'NOT_FOUND'}"
        )

    print()


def print_final_result(
    final_result: dict[str, Any],
) -> None:

    print_separator()

    print(
        "                    FINAL CONSOLIDATED RESULT"
    )

    print_separator()

    print()

    print(
        "Serial Number : "
        f"{final_result.get('serial_number') or 'NOT_FOUND'}"
    )

    print(
        "IMEI          : "
        f"{final_result.get('imei') or 'NOT_FOUND'}"
    )

    print()

    print_separator()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if len(sys.argv) < 2:

        print(
            'Usage: python -m scripts.test_paddleocr '
            '"data/images/dm_2.png"'
        )

        raise SystemExit(1)

    image_path = Path(
        sys.argv[1]
    )

    if not image_path.exists():

        print(
            f"Image not found: {image_path}"
        )

        raise SystemExit(1)

    print(
        "Loading PaddleOCR..."
    )

    backend = PaddleOCRBackend()

    extractor = UniversalOCRExtractor()

    consolidator = (
        UniversalOCRConsolidator()
    )

    try:

        backend.load()

        print(
            "Model loaded."
        )

        print()

        print(
            "Running single cropped meter OCR..."
        )

        print()

        processed_input = (
            backend.preprocess(
                image_path
            )
        )

        prediction = backend.predict(
            processed_input
        )

        candidates = prediction.get(
            "candidates",
            [],
        )

        raw_outputs = {
            candidate["region"]:
                candidate["output"]
            for candidate in candidates
        }

        # --------------------------------------------------
        # Stage 1: Raw PaddleOCR output
        # --------------------------------------------------

        print_raw_outputs(
            raw_outputs
        )

        # --------------------------------------------------
        # Stage 2: Python extraction
        # --------------------------------------------------

        extracted_results = {}

        for region_name, raw_output in (
            raw_outputs.items()
        ):

            extracted_results[
                region_name
            ] = extractor.extract(
                raw_output
            )

        print_extraction_results(
            extracted_results
        )

        # --------------------------------------------------
        # Stage 3: Consolidation
        # --------------------------------------------------

        final_result = (
            consolidator.consolidate(
                extracted_results
            )
        )

        print_final_result(
            final_result
        )

    finally:

        backend.unload()

        print()

        print(
            "Model unloaded."
        )


if __name__ == "__main__":
    main()