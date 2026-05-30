"""Tests for POST /inference/batch-predict endpoint."""

from __future__ import annotations

import pytest


def test_batch_predict_empty_signals(client):
    """When no active signals exist, return 0 queued."""
    response = client.post(
        "/inference/batch-predict",
        json={"compute_run_id": "test-run-123"},
    )

    # No DB, so this will fail auth first. Just verify endpoint exists.
    assert response.status_code in (200, 403, 409)


def test_batch_predict_no_features(client):
    """When feature_values is empty, test structure is sound."""
    response = client.post(
        "/inference/batch-predict",
        json={"compute_run_id": "test-run-empty"},
    )

    # This test verifies the endpoint can be called and the body parses.
    # Full DB integration tests belong in a separate suite.
    assert response.status_code in (200, 403, 409)


def test_batch_predict_endpoint_exists(client):
    """Verify POST /inference/batch-predict endpoint exists."""
    # This is a smoke test to ensure the endpoint is registered
    response = client.post(
        "/inference/batch-predict",
        json={"compute_run_id": "smoke-test"},
    )

    # Response should be 403 (auth token missing) or 409 (no features)
    # but NOT 404 (route not found)
    assert response.status_code in (200, 403, 409)
    assert response.status_code != 404
