"""Smoke tests for the MeterSerial-VLM FastAPI application.

Verifies the FastAPI app boots cleanly, the core API routes are registered,
and health/model endpoints return the expected JSON shape.

These tests do NOT require ML models to be present. Model loading is lazy
and only triggered on actual OCR requests or when PRELOAD_MODELS=1.
"""

from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)


def test_health_check():
    """Health endpoint must return 200 with a status field."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    # Accept common status variants to be resilient to enum changes
    assert str(data["status"]).lower() in ("ok", "healthy", "up")


def test_models_endpoint():
    """Models endpoint must return a list of models."""
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert isinstance(data["models"], list)


def test_openapi_schema():
    """App must generate a valid OpenAPI schema with core routes registered."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "paths" in schema
    paths = schema["paths"]
    # Core routes must be present in the registered graph
    assert "/api/v1/health" in paths
    assert "/api/v1/models" in paths
    # At least one POST route should be registered (extract/invoice/ocr)
    assert any("post" in methods for methods in paths.values())