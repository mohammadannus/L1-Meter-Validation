"""
Health and models routes.
"""

from __future__ import annotations

from fastapi import APIRouter

from api.schemas import HealthResponse, ModelsResponse, QueueStatusResponse
from api.inference import get_available_models, get_default_model, get_current_model_name
from api.ocr_engine import OCREngine
from api.concurrency import get_ocr_guard

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["System"],
)
async def health_check() -> HealthResponse:
    """Returns service health status, API version, active compute device, and loaded model."""
    device = "cpu"
    try:
        import paddle
        if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
            device = "cuda"
    except Exception:
        pass

    ocr_engine = OCREngine.get_instance()
    guard = get_ocr_guard()

    return HealthResponse(
        status="ok",
        version="1.0.0",
        device=device,
        loaded_model=get_current_model_name(),
        ocr_engine_loaded=ocr_engine.is_loaded,
        queue_status=guard.get_status(),
    )


@router.get(
    "/models",
    response_model=ModelsResponse,
    summary="List available models",
    tags=["System"],
)
async def list_models() -> ModelsResponse:
    """Returns the list of available model backends and the default model."""
    return ModelsResponse(
        models=get_available_models(),
        default_model=get_default_model(),
    )


@router.get(
    "/queue/status",
    response_model=QueueStatusResponse,
    summary="Get OCR concurrency queue status",
    tags=["System"],
)
async def get_queue_status() -> QueueStatusResponse:
    """Returns real-time queue depth, active worker count, and processing statistics."""
    guard = get_ocr_guard()
    return QueueStatusResponse(**guard.get_status())

