"""Predictor unit tests with a tiny in-process LightGBM model.

We train a 2-row 2-feature binary classifier inline, then verify the
predict_single / predict_batch contract: feature_vector ordering,
missing-feature error path, and batch skip behavior on partial misses.
"""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pytest

from blackheart_inference.services.predictor import (
    EmptyModelError,
    MissingFeatureError,
    predict_batch,
    predict_single,
)


def _make_tiny_binary_payload() -> dict[str, Any]:
    """Train a 4-row binary classifier on two features. Deterministic seed."""
    X = np.array(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float64
    )
    y = np.array([0, 0, 1, 1], dtype=np.int32)
    booster = lgb.train(
        {
            "objective": "binary",
            "metric": "binary_logloss",
            "num_leaves": 2,
            "learning_rate": 0.1,
            "min_data_in_leaf": 1,
            "min_data_in_bin": 1,
            "verbose": -1,
            "seed": 42,
        },
        lgb.Dataset(X, label=y),
        num_boost_round=5,
    )
    return {
        "content_sha256": "x" * 64,
        "spec": {"name": "test_binary"},
        "feature_names": ["f0", "f1"],
        "booster": booster,
        "ensemble": None,
        "objective": "binary",
    }


def test_predict_single_returns_scalar_probability() -> None:
    payload = _make_tiny_binary_payload()
    value, confidence = predict_single(payload, {"f0": 1.0, "f1": 1.0})
    assert isinstance(value, float)
    assert 0.0 <= value <= 1.0
    assert confidence is None


def test_predict_single_respects_feature_order() -> None:
    """Swapping feature values yields different predictions only if order
    matters — i.e. the predictor is honoring feature_names order, not
    dict iteration order."""
    payload = _make_tiny_binary_payload()
    # Vector A: f0=1, f1=0 (class 1 in training)
    a, _ = predict_single(payload, {"f0": 1.0, "f1": 0.0})
    # Vector B: f0=0, f1=1 (class 0 in training)
    b, _ = predict_single(payload, {"f0": 0.0, "f1": 1.0})
    assert a > b


def test_predict_single_missing_feature_raises_missing_subclass() -> None:
    payload = _make_tiny_binary_payload()
    with pytest.raises(MissingFeatureError) as exc_info:
        predict_single(payload, {"f0": 1.0})  # f1 missing
    assert exc_info.value.missing == ["f1"]
    # Still a ValueError for any caller using the broad except.
    assert isinstance(exc_info.value, ValueError)


def test_predict_single_refuses_empty_booster_with_empty_subclass() -> None:
    payload = {
        "spec": {"name": "empty"},
        "feature_names": ["f0"],
        "booster": None,
        "ensemble": None,
        "objective": "binary",
    }
    with pytest.raises(EmptyModelError):
        predict_single(payload, {"f0": 1.0})


def test_missing_and_empty_are_distinct_subclasses() -> None:
    # The endpoint dispatches HTTP status off these; a regression that
    # collapsed them to bare ValueError would re-introduce the original
    # misclassification bug (corrupt artifact reported as retryable
    # feature_value_missing).
    assert issubclass(MissingFeatureError, ValueError)
    assert issubclass(EmptyModelError, ValueError)
    assert not issubclass(MissingFeatureError, EmptyModelError)
    assert not issubclass(EmptyModelError, MissingFeatureError)


def test_predict_single_refuses_ensemble() -> None:
    payload = {
        "spec": {"name": "ens"},
        "feature_names": ["f0"],
        "booster": None,
        "ensemble": {"placeholder": True},
        "objective": "binary",
    }
    with pytest.raises(NotImplementedError):
        predict_single(payload, {"f0": 1.0})


def test_predict_batch_skips_rows_with_missing_features() -> None:
    payload = _make_tiny_binary_payload()
    rows = [
        ("t0", {"f0": 1.0, "f1": 1.0}),
        ("t1", {"f0": 1.0}),                  # missing f1 → None
        ("t2", {"f0": 0.0, "f1": 0.0}),
    ]
    out = predict_batch(payload, rows)
    assert len(out) == 3
    assert out[0][1] is not None
    assert out[1][1] is None
    assert out[2][1] is not None


def test_predict_batch_empty_input() -> None:
    payload = _make_tiny_binary_payload()
    assert predict_batch(payload, []) == []
