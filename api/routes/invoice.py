"""
Invoice extraction endpoint — accepts an uploaded invoice/document photo
and returns Application No. and Account Id.
"""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
import time
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from api.ocr_engine import OCREngine
from api.concurrency import get_ocr_guard
from api.schemas import InvoiceExtractResponse
from utils.invoice_extractor import InvoiceExtractor

logger = logging.getLogger(__name__)
router = APIRouter()

# Shared invoice extractor instance
_invoice_extractor = InvoiceExtractor()

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


def _run_invoice_ocr(image_path: Path) -> dict:
    """
    Run OCR on an invoice image and extract Application No.

    This runs in a worker thread (called via asyncio.to_thread).
    """
    start_time = time.perf_counter()

    engine = OCREngine.get_instance()

    # For invoices, we don't need YOLO meter detection —
    # run OCR directly on the full document image
    raw_text = engine.run_ocr_from_path(
        image_path,
        max_dimension=2048,  # Invoices are text-heavy, allow slightly larger
        min_confidence=0.30,
    )

    # Extract invoice-specific fields
    extraction = _invoice_extractor.extract(raw_text)

    # If Application No. was not found, retry with contrast enhancement.
    # Utility documents photographed on tables often suffer from shadows or faint ink.
    if not extraction.get("application_no"):
        try:
            from PIL import Image, ImageEnhance
            with Image.open(image_path) as img:
                rgb_img = img.convert("RGB")
                enhanced = ImageEnhance.Contrast(rgb_img).enhance(1.6)
                enhanced_text = engine.run_ocr(enhanced, min_confidence=0.25)
                retry_ext = _invoice_extractor.extract(enhanced_text)
                if retry_ext.get("application_no"):
                    extraction = retry_ext
                    raw_text = enhanced_text
                elif retry_ext.get("account_id") and not extraction.get("account_id"):
                    extraction["account_id"] = retry_ext["account_id"]
        except Exception as retry_err:
            logger.debug("Contrast-enhanced OCR retry failed: %s", retry_err)

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return {
        "application_no": extraction.get("application_no", ""),
        "account_id": extraction.get("account_id", ""),
        "reference_no": extraction.get("reference_no", ""),
        "raw_text": extraction.get("raw_text", raw_text),
        "processing_time_ms": round(elapsed_ms, 2),
    }



@router.post(
    "/invoice/extract",
    response_model=InvoiceExtractResponse,
    summary="Extract Application No. from an invoice document photo",
    tags=["Invoice"],
    responses={
        400: {"description": "Invalid file"},
        503: {"description": "Server busy — queue full"},
        504: {"description": "Request timed out in queue"},
        500: {"description": "Inference error"},
    },
)
async def invoice_extract(
    file: UploadFile = File(
        ...,
        description="Invoice/document photo (JPEG, PNG, WEBP, BMP, TIFF)",
    ),
) -> InvoiceExtractResponse:
    """
    Upload an invoice or procurement letter photo to extract:
    - **application_no** — the Application Number from the document
    - **account_id** — the Account ID if present
    - **reference_no** — the BESCOM reference number if present
    - **raw_text** — full OCR text for verification

    The endpoint uses the shared PaddleOCR engine with concurrency protection.
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

        # Run OCR + extraction in worker thread
        result = await asyncio.to_thread(_run_invoice_ocr, tmp_path)

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception(f"Invoice extraction failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Invoice extraction failed: {exc}",
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
    response_payload = InvoiceExtractResponse(
        application_no=result.get("application_no", ""),
        account_id=result.get("account_id", ""),
        reference_no=result.get("reference_no", ""),
        raw_text=result.get("raw_text", ""),
        processing_time_ms=result.get("processing_time_ms", 0.0),
    )

    # ------------------------------------------------------------------
    # Print extraction output to terminal
    # ------------------------------------------------------------------
    try:
        print("\n" + "=" * 64, flush=True)
        print(
            f"  [INVOICE EXTRACTION RESULT]  |  {file.filename or 'uploaded_image'}",
            flush=True,
        )
        print("=" * 64, flush=True)
        print(
            f"  Application No : {response_payload.application_no or 'NOT_FOUND'}",
            flush=True,
        )
        print(
            f"  Account ID     : {response_payload.account_id or 'NOT_FOUND'}",
            flush=True,
        )
        print(
            f"  Reference No   : {response_payload.reference_no or 'NOT_FOUND'}",
            flush=True,
        )
        print(
            f"  Time Taken     : {response_payload.processing_time_ms:.2f} ms",
            flush=True,
        )
        print("-" * 64, flush=True)
        print("  RAW JSON PAYLOAD:", flush=True)
        print(json.dumps(response_payload.model_dump(), indent=2), flush=True)
        print("=" * 64 + "\n", flush=True)
    except Exception as print_err:
        logger.warning(f"Could not print output to terminal: {print_err}")

    return response_payload
