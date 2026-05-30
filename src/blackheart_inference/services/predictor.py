"""Run ``Booster.predict()`` on a single feature vector or a batch.

Booster prediction semantics (per blackheart-train's artifact doc):

  * ``objective='binary'``      → ``predict()`` returns class-1 probabilities
  * ``objective='regression'``  → ``predict()`` returns raw forecasts
  * ``objective='multiclass'``  → softmax probabilities per class

Inference only handles single-model artifacts in this skeleton — the
ensemble path (payload['ensemble'] populated) is a follow-up. The
artifact loader still tolerates v2 ensemble payloads at the read level;
this module refuses them at the predict level so callers get a clear
error rather than a None booster.

Two distinct ``ValueError`` subclasses so callers can dispatch HTTP
status correctly: a missing feature is a transient research-state
problem (run a /features backfill and retry → 409 retryable), but an
empty booster is a corrupted artifact (502 not retryable).
"""

from __future__ import annotations

from typing import Any

import numpy as np


class MissingFeatureError(ValueError):
    """One or more model inputs were absent from the feature_vector.

    Caller should run /features/{name}/v/{version}/backfill to populate
    the missing rows and retry. Retryable.
    """

    def __init__(self, missing: list[str]) -> None:
        self.missing = list(missing)
        head = self.missing[:10]
        tail = "..." if len(self.missing) > 10 else ""
        super().__init__(
            f"feature vector missing {len(self.missing)} required input(s): "
            f"{head}{tail}"
        )


class EmptyModelError(ValueError):
    """Artifact has neither a booster nor an ensemble — corrupted payload.

    The artifact_loader already verified content_sha; reaching this
    branch means blackheart-train wrote an artifact whose model body
    didn't survive serialisation. Not retryable; the artifact must be
    re-trained.
    """


def predict_single(
    payload: dict[str, Any],
    feature_vector: dict[str, float | None],
) -> tuple[float, float | None]:
    """Return ``(value, confidence)`` from one feature row.

    For binary: ``value`` is P(class=1), ``confidence`` is None (the
    JVM's MLRegimeGateGuard already reasons about distance from 0.5).
    For regression: ``value`` is the raw forecast, ``confidence`` is None.

    Raises ``ValueError`` when a required feature is missing from the
    vector (None in feature_vector). We do NOT silently impute — the
    JVM consumer has a fail-open contract and we honour it by surfacing
    the gap to the caller, which decides whether to skip writing the
    row entirely.
    """
    if payload.get("ensemble") is not None:
        raise NotImplementedError(
            "Ensemble artifacts not yet supported by this skeleton. "
            "The artifact format is read correctly; the predictor "
            "intentionally refuses to run partial ensembles."
        )
    booster = payload.get("booster")
    if booster is None:
        raise EmptyModelError(
            "Artifact payload has no booster and no ensemble — refusing "
            "to predict against an empty model."
        )

    feature_names: list[str] = list(payload["feature_names"])
    row: list[float] = []
    missing: list[str] = []
    for name in feature_names:
        v = feature_vector.get(name)
        if v is None:
            missing.append(name)
        else:
            row.append(float(v))
    if missing:
        raise MissingFeatureError(missing)

    X = np.asarray([row], dtype=np.float64)
    raw = booster.predict(X)
    # LightGBM returns 1-D ndarray for binary/regression, 2-D for multiclass.
    arr = np.asarray(raw)
    if arr.ndim == 1:
        value = float(arr[0])
    elif arr.ndim == 2:
        # Multiclass — collapse to the argmax-class probability so downstream
        # consumers see a scalar. Forward-compat: a future spec can stash
        # the per-class vector in signal_history.meta.
        value = float(arr[0].max())
    else:
        raise ValueError(f"unexpected predict() shape: {arr.shape}")

    return value, None


def predict_batch(
    payload: dict[str, Any],
    rows: list[tuple[Any, dict[str, float | None]]],
) -> list[tuple[Any, float | None]]:
    """Batch predict over ``[(ts, feature_vector), ...]``.

    Skips rows with any missing feature, returning ``(ts, None)`` for
    them so the caller can decide whether to upsert a row (typically NO
    — we don't fabricate signals to fill gaps). Empty input → empty output.
    """
    if not rows:
        return []
    if payload.get("ensemble") is not None:
        raise NotImplementedError("Ensemble artifacts not yet supported.")
    booster = payload.get("booster")
    if booster is None:
        raise EmptyModelError("Artifact has no booster.")

    feature_names: list[str] = list(payload["feature_names"])
    valid_idx: list[int] = []
    valid_X: list[list[float]] = []
    out: list[tuple[Any, float | None]] = []
    for i, (ts, vec) in enumerate(rows):
        row = []
        any_missing = False
        for name in feature_names:
            v = vec.get(name)
            if v is None:
                any_missing = True
                break
            row.append(float(v))
        if any_missing:
            out.append((ts, None))
        else:
            valid_idx.append(i)
            valid_X.append(row)
            out.append((ts, None))  # placeholder, fill below

    if valid_X:
        X = np.asarray(valid_X, dtype=np.float64)
        raw = booster.predict(X)
        arr = np.asarray(raw)
        if arr.ndim == 2:
            preds = arr.max(axis=1)
        else:
            preds = arr
        for idx, val in zip(valid_idx, preds, strict=True):
            ts, _ = out[idx]
            out[idx] = (ts, float(val))
    return out
