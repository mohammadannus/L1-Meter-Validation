"""
Pydantic schemas for the MeterSerial-VLM API.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RegionResult(BaseModel):
    """OCR result for a single image region."""

    region: str = Field(..., description="Region name (e.g. full_image, tile_1)")
    raw_output: str = Field(..., description="Raw text output from the OCR model")
    serial_number: str = Field("", description="Extracted serial number for this region")
    imei: str = Field("", description="Extracted IMEI for this region")


class SealDetail(BaseModel):
    """Detection detail for an individual meter seal."""

    confidence: float = Field(..., description="Detection confidence score for this seal (0.0 to 1.0)")
    box: list[float] = Field(..., description="Bounding box [x1, y1, x2, y2] coordinates")
    class_name: str = Field("seal", description="Class name of the detected object")


class ExtractResponse(BaseModel):
    """Response from the /extract endpoint."""

    serial_number: str = Field("", description="Final consolidated serial number")
    imei: str = Field("", description="Final consolidated IMEI")
    green_light_status: str = Field(
        "unknown",
        description="Power indicator status from green LED: 'powered_on', 'powered_off', or 'unknown'",
    )
    green_light_percent: float = Field(
        0.0,
        description="Percentage of green pixels detected in the Display region",
    )
    power_status: str = Field(
        "unknown",
        description="Meter power status: 'powered_on' (green light present), 'powered_off' (green light absent), or 'unknown'",
    )
    seal_count: int = Field(
        0,
        description="Total number of physical seals identified on the meter",
    )
    meter_locked: bool = Field(
        False,
        description="Indicates whether meter is locked (True if seals are detected, False otherwise)",
    )
    seal_confidences: list[float] = Field(
        default_factory=list,
        description="Confidence scores for all detected seals",
    )
    average_seal_confidence: float = Field(
        0.0,
        description="Average confidence across all detected seals",
    )
    seals: list[SealDetail] = Field(
        default_factory=list,
        description="Per-seal detection details including bounding boxes and confidence scores",
    )
    model_used: str = Field(..., description="Model backend used for inference")
    processing_time_ms: float = Field(..., description="Total inference time in milliseconds")
    regions: list[RegionResult] = Field(
        default_factory=list,
        description="Per-region extraction details",
    )


# =========================================================
# Invoice extraction response
# =========================================================

class InvoiceExtractResponse(BaseModel):
    """Response from the /invoice/extract endpoint."""

    application_no: str = Field(
        "",
        description="Application Number extracted from the invoice document",
    )
    account_id: str = Field(
        "",
        description="Account ID extracted from the invoice document",
    )
    reference_no: str = Field(
        "",
        description="BESCOM reference number if found in the document",
    )
    raw_text: str = Field(
        "",
        description="Full OCR text from the document for verification",
    )
    processing_time_ms: float = Field(
        ...,
        description="Total processing time in milliseconds",
    )


# =========================================================
# Generic OCR response
# =========================================================

class OCRExtractResponse(BaseModel):
    """Response from the /ocr/extract endpoint."""

    raw_text: str = Field(
        "",
        description="All recognized text, newline-separated",
    )
    lines: list[str] = Field(
        default_factory=list,
        description="Individual text lines detected by OCR",
    )
    line_count: int = Field(
        0,
        description="Total number of text lines detected",
    )
    processing_time_ms: float = Field(
        ...,
        description="Total processing time in milliseconds",
    )


# =========================================================
# Queue status response
# =========================================================

class QueueStatusResponse(BaseModel):
    """Response from the /queue/status endpoint."""

    max_concurrent: int = Field(..., description="Maximum parallel OCR inferences allowed")
    max_queue_depth: int = Field(..., description="Maximum requests allowed to wait in queue")
    active_requests: int = Field(..., description="Number of OCR inferences currently running")
    queued_requests: int = Field(..., description="Number of requests currently waiting in queue")
    total_processed: int = Field(..., description="Total requests processed since server start")
    total_rejected: int = Field(..., description="Total requests rejected (queue full / timeout)")
    request_timeout_sec: float = Field(..., description="Seconds before a queued request times out")


# =========================================================
# Health / Models / Error responses
# =========================================================

class HealthResponse(BaseModel):
    """Response from the /health endpoint."""

    status: str = Field("ok", description="Service health status")
    version: str = Field("1.0.0", description="API version")
    device: str = Field("cpu", description="Compute device currently active (cpu / cuda)")
    loaded_model: str | None = Field(None, description="Currently loaded model in memory")
    ocr_engine_loaded: bool = Field(False, description="Whether the shared OCR engine is loaded")
    queue_status: dict[str, Any] | None = Field(
        None,
        description="Current OCR request queue status",
    )


class ModelsResponse(BaseModel):
    """Response from the /models endpoint."""

    models: list[str] = Field(..., description="Available model backend names")
    default_model: str = Field(..., description="Default model used when none specified")


class ErrorResponse(BaseModel):
    """Standard error response."""

    detail: str = Field(..., description="Error description")
