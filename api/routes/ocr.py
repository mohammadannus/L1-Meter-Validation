"""
Generic OCR text extraction endpoint — accepts any image or document photo
and returns the raw OCR text for future use cases.
"""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
import time
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from api.ocr_engine import OCREngine
from api.concurrency import get_ocr_guard
from api.schemas import OCRExtractResponse

logger = logging.getLogger(__name__)
router = APIRouter()

# Accepted image MIME types
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/tiff",
}

# Max upload size: 20 MB
MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024


def _run_generic_ocr(image_path: Path) -> dict:
    """
    Run OCR on any image and return raw text.

    This runs in a worker thread (called via asyncio.to_thread).
    """
    start_time = time.perf_counter()

    engine = OCREngine.get_instance()

    # Direct OCR — no domain-specific preprocessing
    raw_text = engine.run_ocr_from_path(
        image_path,
        max_dimension=2048,
        min_confidence=0.25,  # Slightly lower threshold for general use
    )

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    lines = [line for line in raw_text.split("\n") if line.strip()]

    return {
        "raw_text": raw_text,
        "line_count": len(lines),
        "lines": lines,
        "processing_time_ms": round(elapsed_ms, 2),
    }


@router.post(
    "/ocr/extract",
    response_model=OCRExtractResponse,
    summary="Extract raw text from any image or document photo",
    tags=["OCR"],
    responses={
        400: {"description": "Invalid file"},
        503: {"description": "Server busy — queue full"},
        504: {"description": "Request timed out in queue"},
        500: {"description": "OCR error"},
    },
)
async def ocr_extract(
    file: UploadFile = File(
        ...,
        description="Image or document photo (JPEG, PNG, WEBP, BMP, TIFF)",
    ),
) -> OCRExtractResponse:
    """
    Upload any image or document photo to extract raw text using PaddleOCR.

    This is a **generic OCR endpoint** designed for future use cases —
    it returns the full OCR text without any domain-specific extraction.

    Response includes:
    - **raw_text** — all recognized text (newline-separated)
    - **lines** — individual text lines as an array
    - **line_count** — total number of text lines detected
    """

    # ------------------------------------------------------------------
    # Validate content type
    # ------------------------------------------------------------------
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file type '{file.content_type}'. "
                f"Accepted types: {', '.join(sorted(ALLOWED_CONTENT_TYPES))}"
            ),
        )

    # ------------------------------------------------------------------
    # Read file and enforce size limit
    # ------------------------------------------------------------------
    file_bytes = await file.read()

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"File too large ({len(file_bytes) / 1024 / 1024:.1f} MB). "
                f"Maximum allowed size is {MAX_FILE_SIZE_BYTES // 1024 // 1024} MB."
            ),
        )

    # ------------------------------------------------------------------
    # Acquire concurrency slot
    # ------------------------------------------------------------------
    guard = get_ocr_guard()
    await guard.acquire()

    suffix = Path(file.filename or "image.jpg").suffix or ".jpg"
    tmp_path: Path | None = None

    try:
        # Write to temp file
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = Path(tmp.name)

        # Run OCR in worker thread
        result = await asyncio.to_thread(_run_generic_ocr, tmp_path)

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception(f"Generic OCR extraction failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OCR extraction failed: {exc}",
        ) from exc

    finally:
        # Always release the concurrency slot
        guard.release()

        # Always clean up the temp file
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Build response
    # ------------------------------------------------------------------
    response_payload = OCRExtractResponse(
        raw_text=result.get("raw_text", ""),
        lines=result.get("lines", []),
        line_count=result.get("line_count", 0),
        processing_time_ms=result.get("processing_time_ms", 0.0),
    )

    # ------------------------------------------------------------------
    # Print extraction output to terminal
    # ------------------------------------------------------------------
    try:
        print("\n" + "=" * 64, flush=True)
        print(
            f"  [GENERIC OCR RESULT]  |  {file.filename or 'uploaded_image'}",
            flush=True,
        )
        print("=" * 64, flush=True)
        print(
            f"  Lines Detected : {response_payload.line_count}",
            flush=True,
        )
        print(
            f"  Time Taken     : {response_payload.processing_time_ms:.2f} ms",
            flush=True,
        )
        print("-" * 64, flush=True)
        # Show first 10 lines of text
        preview_lines = response_payload.lines[:10]
        for line in preview_lines:
            print(f"    {line}", flush=True)
        if response_payload.line_count > 10:
            print(
                f"    ... ({response_payload.line_count - 10} more lines)",
                flush=True,
            )
        print("=" * 64 + "\n", flush=True)
    except Exception as print_err:
        logger.warning(f"Could not print output to terminal: {print_err}")

    return response_payload
