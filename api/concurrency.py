"""
Concurrency guard for OCR inference requests.

Implements a bounded async semaphore with queue depth tracking to:
1. Limit parallel OCR inference (prevents OOM)
2. Cap the waiting queue (prevents request starvation)
3. Timeout long-waiting requests (prevents indefinite hangs)

Usage in route handlers:

    guard = get_ocr_guard()
    await guard.acquire()
    try:
        result = await asyncio.to_thread(run_inference, ...)
    finally:
        guard.release()
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading

from fastapi import HTTPException, status

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (from env or defaults)
# ---------------------------------------------------------------------------

def _env_int(key: str, default: int) -> int:
    """Read an integer from environment, falling back to default."""
    try:
        return int(os.environ.get(key, str(default)))
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# Concurrency Guard
# ---------------------------------------------------------------------------

class OCRConcurrencyGuard:
    """
    Bounded semaphore + queue depth tracking for OCR inference.

    Parameters
    ----------
    max_concurrent : int
        Maximum number of OCR inferences running simultaneously.
    max_queue_depth : int
        Maximum number of requests allowed to wait in the queue.
        Requests beyond this are rejected with HTTP 503.
    request_timeout : float
        Seconds to wait for a semaphore slot before timing out (HTTP 504).
    """

    def __init__(
        self,
        max_concurrent: int | None = None,
        max_queue_depth: int | None = None,
        request_timeout: float | None = None,
    ) -> None:
        self.max_concurrent = max_concurrent or _env_int("MAX_CONCURRENT_OCR", 2)
        self.max_queue_depth = max_queue_depth or _env_int("MAX_QUEUE_DEPTH", 20)
        self.request_timeout = request_timeout or float(
            _env_int("OCR_REQUEST_TIMEOUT", 120)
        )

        self._semaphore = asyncio.Semaphore(self.max_concurrent)
        self._queue_depth = 0
        self._active_count = 0
        self._total_processed = 0
        self._total_rejected = 0
        self._lock = threading.Lock()

        logger.info(
            "OCRConcurrencyGuard initialized: "
            "max_concurrent=%d, max_queue=%d, timeout=%ds",
            self.max_concurrent,
            self.max_queue_depth,
            self.request_timeout,
        )

    # -----------------------------------------------------------------
    # Acquire / Release
    # -----------------------------------------------------------------

    async def acquire(self) -> None:
        """
        Acquire a semaphore slot for OCR inference.

        Raises
        ------
        HTTPException(503)
            If the queue is full (too many requests waiting).
        HTTPException(504)
            If the request waited too long for a slot.
        """
        # Check queue capacity
        with self._lock:
            if self._queue_depth >= self.max_queue_depth:
                self._total_rejected += 1
                logger.warning(
                    "OCR queue full (%d/%d). Rejecting request.",
                    self._queue_depth,
                    self.max_queue_depth,
                )
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=(
                        f"Server is busy processing {self._active_count} requests "
                        f"with {self._queue_depth} waiting. Please retry shortly."
                    ),
                    headers={"Retry-After": "5"},
                )
            self._queue_depth += 1

        # Wait for a semaphore slot (with timeout)
        try:
            await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=self.request_timeout,
            )
        except asyncio.TimeoutError:
            with self._lock:
                self._queue_depth -= 1
                self._total_rejected += 1
            logger.warning(
                "OCR request timed out after %ds in queue.", self.request_timeout
            )
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=(
                    f"Request timed out after {self.request_timeout}s "
                    f"waiting for an available OCR processing slot."
                ),
            )

        # Slot acquired — move from queue to active
        with self._lock:
            self._queue_depth -= 1
            self._active_count += 1

    def release(self) -> None:
        """Release the semaphore slot after OCR inference completes."""
        with self._lock:
            self._active_count -= 1
            self._total_processed += 1
        self._semaphore.release()

    # -----------------------------------------------------------------
    # Status
    # -----------------------------------------------------------------

    def get_status(self) -> dict:
        """Return current queue/concurrency status for health endpoint."""
        with self._lock:
            return {
                "max_concurrent": self.max_concurrent,
                "max_queue_depth": self.max_queue_depth,
                "active_requests": self._active_count,
                "queued_requests": self._queue_depth,
                "total_processed": self._total_processed,
                "total_rejected": self._total_rejected,
                "request_timeout_sec": self.request_timeout,
            }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_guard: OCRConcurrencyGuard | None = None
_guard_lock = threading.Lock()


def get_ocr_guard() -> OCRConcurrencyGuard:
    """Get or create the global OCR concurrency guard."""
    global _guard
    if _guard is None:
        with _guard_lock:
            if _guard is None:
                _guard = OCRConcurrencyGuard()
    return _guard


def init_ocr_guard(
    max_concurrent: int | None = None,
    max_queue_depth: int | None = None,
    request_timeout: float | None = None,
) -> OCRConcurrencyGuard:
    """Initialize the global OCR concurrency guard with custom settings."""
    global _guard
    with _guard_lock:
        _guard = OCRConcurrencyGuard(
            max_concurrent=max_concurrent,
            max_queue_depth=max_queue_depth,
            request_timeout=request_timeout,
        )
    return _guard
