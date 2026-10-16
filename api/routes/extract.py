"""
Extract endpoint — accepts an uploaded image and returns Serial Number + IMEI.
"""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from api.inference import DEFAULT_MODEL, AVAILABLE_MODELS, run_inference
from api.concurrency import get_ocr_guard
from api.schemas import ExtractResponse, RegionResult

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


@router.post(
    "/extract",
    response_model=ExtractResponse,
    summary="Extract Serial Number and IMEI from a meter image",
    tags=["Inference"],
    responses={
        400: {"description": "Invalid file or model name"},
        422: {"description": "Validation error"},
        500: {"description": "Inference error"},
    },
)
async def extract(
    file: UploadFile = File(..., description="Meter image file (JPEG, PNG, WEBP, BMP, TIFF)"),
    model: str = Form(
        default=DEFAULT_MODEL,
        description=f"Model backend to use. One of: {', '.join(AVAILABLE_MODELS)}",
    ),
) -> ExtractResponse:
    """
    Upload a meter image and extract:
    - **serial_number** — the meter's serial / SL number
    - **imei** — the meter's IMEI (cellular module identifier)

    The response includes per-region details showing exactly how each
    OCR region contributed to the final result.
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
    # Validate model name
    # ------------------------------------------------------------------
    if model not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unknown model '{model}'. "
                f"Available models: {', '.join(AVAILABLE_MODELS)}"
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
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = Path(tmp.name)

        # Run CPU/GPU-heavy inference in a separate worker thread to avoid blocking the event loop
        result = await asyncio.to_thread(run_inference, image_path=tmp_path, model_name=model)

    except HTTPException:
        raise

    except ValueError as exc:
        logger.warning(f"Validation error in extract: {exc}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception(f"Inference failed with exception: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {exc}",
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
    regions = [
        RegionResult(
            region=r["region"],
            raw_output=r["raw_output"],
            serial_number=r.get("serial_number", ""),
            imei=r.get("imei", ""),
        )
        for r in result.get("regions", [])
    ]

    response_payload = ExtractResponse(
        serial_number=result.get("serial_number", ""),
        imei=result.get("imei", ""),
        green_light_status=result.get("green_light_status", "unknown"),
        green_light_percent=result.get("green_light_percent", 0.0),
        power_status=result.get("power_status", result.get("green_light_status", "unknown")),
        seal_count=result.get("seal_count", 0),
        meter_locked=result.get("meter_locked", False),
        seal_confidences=result.get("seal_confidences", []),
        average_seal_confidence=result.get("average_seal_confidence", 0.0),
        seals=result.get("seals", []),
        model_used=result.get("model_used", model),
        processing_time_ms=result.get("processing_time_ms", 0.0),
        regions=regions,
    )

    # ------------------------------------------------------------------
    # Print extraction output to terminal
    # ------------------------------------------------------------------
    try:
        print("\n" + "=" * 64, flush=True)
        print(f"  [METER EXTRACTION RESULT]  |  {file.filename or 'uploaded_image'}", flush=True)
        print("=" * 64, flush=True)
        print(f"  Serial Number  : {response_payload.serial_number or 'NOT_FOUND'}", flush=True)
        print(f"  IMEI           : {response_payload.imei or 'NOT_FOUND'}", flush=True)

        is_powered_on = response_payload.power_status == "powered_on" or response_payload.green_light_status in ("powered_on", "connected")
        is_powered_off = response_payload.power_status == "powered_off" or response_payload.green_light_status in ("powered_off", "disconnected")
        power_icon = "[ON] " if is_powered_on else "[OFF]"
        power_label = "POWERED ON" if is_powered_on else "POWERED OFF" if is_powered_off else "UNKNOWN"
        print(f"  Power Status   : {power_icon} {power_label} ({response_payload.green_light_percent}% green)", flush=True)

        lock_icon = "[LOCKED]  " if response_payload.meter_locked else "[UNLOCKED]"
        print(f"  Meter Lock     : {lock_icon} {response_payload.seal_count} seal(s) detected (conf: {response_payload.seal_confidences})", flush=True)

        print(f"  Model Used     : {response_payload.model_used}", flush=True)
        print(f"  Time Taken     : {response_payload.processing_time_ms:.2f} ms", flush=True)
        print(f"  Regions Count : {len(response_payload.regions)}", flush=True)
        print("-" * 64, flush=True)
        print("  RAW JSON PAYLOAD:", flush=True)
        print(json.dumps(response_payload.model_dump(), indent=2), flush=True)
        print("=" * 64 + "\n", flush=True)
    except Exception as print_err:
        logger.warning(f"Could not print output to terminal: {print_err}")

    return response_payload
