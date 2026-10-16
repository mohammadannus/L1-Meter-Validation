"""
MeterSerial-VLM  |  Application Entry Point

Run the FastAPI server:
    python main.py

Or directly via uvicorn:
    uvicorn api.app:app --host 0.0.0.0 --port 8000
"""

import os

# Prevent OpenMP runtime conflicts between PyTorch and PaddlePaddle
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Pre-import torch before paddle to ensure safe DLL loading order
try:
    import torch  # noqa: F401
except ImportError:
    pass

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
