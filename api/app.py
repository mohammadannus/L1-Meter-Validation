"""
MeterSerial-VLM FastAPI application.

Serves:
- REST API at /api/v1/
- Interactive docs at /docs and /redoc
- Frontend UI at /
"""

from __future__ import annotations

import os

# Prevent OpenMP runtime conflicts between PyTorch and PaddlePaddle
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Pre-import torch before paddle to ensure safe DLL loading order
try:
    import torch  # noqa: F401
except ImportError:
    pass

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from api.routes.health import router as health_router
from api.routes.extract import router as extract_router
from api.routes.invoice import router as invoice_router
from api.routes.ocr import router as ocr_router
from api.ocr_engine import OCREngine
from api.concurrency import get_ocr_guard


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — runs on startup and shutdown."""
    print("=" * 60)
    print("  MeterSerial-VLM API  |  http://localhost:8000")
    print("  Docs                 |  http://localhost:8000/docs")
    print("=" * 60)
    # Initialize concurrency guard
    guard = get_ocr_guard()
    print(f"  OCR Queue Guard: max_concurrent={guard.max_concurrent}, max_queue={guard.max_queue_depth}")
    if os.getenv("PRELOAD_MODELS", "0") == "1":
        print("  Preloading shared PaddleOCR engine...")
        OCREngine.get_instance().load()
        print("  Shared PaddleOCR engine ready.")
    yield
    print("\nMeterSerial-VLM API shutting down.")



# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="MeterSerial-VLM API",
    description=(
        "L1 Validation API for extracting Serial Numbers and IMEI numbers "
        "from electricity meter images using multi-scale OCR and VLM models."
    ),
    version="1.0.0",
    contact={
        "name": "MeterSerial-VLM",
    },
    license_info={
        "name": "Proprietary",
    },
    lifespan=lifespan,
)

import os
from fastapi.middleware.gzip import GZipMiddleware

# ---------------------------------------------------------------------------
# Middlewares: Compression & CORS
# ---------------------------------------------------------------------------

app.add_middleware(GZipMiddleware, minimum_size=1000)

cors_env = os.getenv("CORS_ORIGINS", "*").strip()
allowed_origins = [o.strip() for o in cors_env.split(",") if o.strip()] if cors_env != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

app.include_router(health_router, prefix="/api/v1")
app.include_router(extract_router, prefix="/api/v1")
app.include_router(invoice_router, prefix="/api/v1")
app.include_router(ocr_router, prefix="/api/v1")

# ---------------------------------------------------------------------------
# Frontend static files
# ---------------------------------------------------------------------------

_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

if _FRONTEND_DIR.exists():
    # Serve static assets (CSS, JS, images)
    app.mount(
        "/static",
        StaticFiles(directory=str(_FRONTEND_DIR)),
        name="static",
    )

    @app.get("/", include_in_schema=False)
    async def serve_index():
        """Serve the frontend UI."""
        return FileResponse(str(_FRONTEND_DIR / "index.html"))
