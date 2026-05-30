"""``/inference/run`` and ``/inference/backfill``.

Researcher-facing surface goes through the orchestrator's proxy at
``:8082/inference/*``. This service is loopback-only — direct callers
are the orchestrator and (eventually) the live-streaming worker cron.

Workflow (run):
  1. Resolve ``signal_id`` → ``signal_definition`` row.
  2. Resolve ``model_id`` → ``model_registry`` row (must have artifact_sha256).
  3. Load the artifact from the local content-addressed store.
  4. Fetch feature versions from ``feature_registry``.
  5. Fetch per-bar values from ``feature_values`` at the requested ts.
  6. Run ``Booster.predict()``.
  7. UPSERT one row into ``signal_history``.

Backfill is the same minus step 5 (range-query) + step 6 (batch) +
step 7 (batched UPSERT).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import UUID

import asyncpg
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from ..errors import InferenceError, NextAction
from ..repo import features as features_repo
from ..repo import models as models_repo
from ..repo import signals as signals_repo
from ..services.artifact_loader import read_artifact
from ..services.predictor import (
    EmptyModelError,
    MissingFeatureError,
    predict_batch,
    predict_single,
)
from .deps import get_agent_name, get_db_conn

router = APIRouter(prefix="/inference", tags=["inference"])

_VALID_INTERVALS = frozenset({"5m", "15m", "1h", "4h"})
_REFUSED_STATUSES = frozenset({"retired", "rejected_by_operator"})
_INTERVAL_SECONDS: dict[str, int] = {"5m": 300, "15m": 900, "1h": 3600, "4h": 14400}


def _estimate_backfill_bars(start: datetime, end: datetime, interval_name: str) -> int:
    """Cheap upper bound on row count from window width + bar cadence.

    Used to refuse oversized windows BEFORE the heavy feature-fetch SQL.
    Overestimates slightly (assumes contiguous bars; real data may have
    gaps) which is the safe direction — never under-counts.
    """
    seconds_per_bar = _INTERVAL_SECONDS[interval_name]
    delta_s = max(0.0, (end - start).total_seconds())
    return int(delta_s // seconds_per_bar) + 1


class RunBody(BaseModel):
    """Single-bar inference. Honors ``Idempotency-Key`` at the orchestrator
    proxy layer; this service writes UPSERT-style so a replay returns
    the same shape regardless."""

    signal_id: UUID
    symbol: str = Field(..., min_length=1, max_length=20)
    ts: datetime
    interval_name: str = Field(..., description="5m|15m|1h|4h")
    source: Literal["stream", "catchup_scan", "historical_replay"] = "stream"


class BackfillBody(BaseModel):
    """Window inference. ``end`` is exclusive (Postgres range conventions)."""

    signal_id: UUID
    symbol: str = Field(..., min_length=1, max_length=20)
    interval_name: str = Field(..., description="5m|15m|1h|4h")
    start: datetime
    end: datetime
    source: Literal["catchup_scan", "historical_replay"] = "historical_replay"


def _validate_interval(interval_name: str) -> None:
    if interval_name not in _VALID_INTERVALS:
        raise InferenceError(
            status_code=400,
            error_code="bad_interval",
            message=f"interval_name={interval_name!r} not in {sorted(_VALID_INTERVALS)}.",
            retryable=False,
        )


async def _resolve_artifact(
    conn: asyncpg.Connection, signal_id: UUID
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve signal → model → artifact. Returns ``(signal_row, model_row)``."""
    signal = await models_repo.fetch_signal_definition(conn, signal_id)
    if signal is None:
        raise InferenceError(
            status_code=404,
            error_code="signal_not_found",
            message=f"signal_id={signal_id} not in signal_definition.",
            retryable=False,
            next_action=NextAction(kind="call", method="POST", path="/models/register"),
        )
    if signal["status"] == "retired":
        raise InferenceError(
            status_code=409,
            error_code="signal_retired",
            message=f"signal_id={signal_id} is retired; refusing to write signal_history.",
            retryable=False,
        )
    model = await models_repo.fetch_model_registry_row(conn, signal["model_id"])
    if model is None:
        raise InferenceError(
            status_code=404,
            error_code="model_not_found",
            message=(
                f"model_registry row {signal['model_id']} not found "
                f"for signal_id={signal_id}."
            ),
            retryable=False,
        )
    if model["status"] in _REFUSED_STATUSES:
        raise InferenceError(
            status_code=409,
            error_code="model_lifecycle_refused",
            message=(
                f"model status={model['status']!r} — refusing to serve. "
                "Acceptable states: training, trained, staged, shadow, "
                "cooling_down, live."
            ),
            retryable=False,
            details={"model_id": str(model["id"]), "status": model["status"]},
        )
    if not model.get("artifact_sha256"):
        raise InferenceError(
            status_code=409,
            error_code="model_artifact_missing",
            message=(
                f"model {model['id']} has no artifact_sha256 — registration "
                "incomplete. Re-run blackheart-train with --register."
            ),
            retryable=False,
        )
    return signal, model


@router.post("/run")
async def run_inference(
    body: RunBody,
    request: Request,
    conn: asyncpg.Connection = Depends(get_db_conn),
    agent: str = Depends(get_agent_name),
) -> dict[str, Any]:
    _validate_interval(body.interval_name)

    signal, model = await _resolve_artifact(conn, body.signal_id)

    # Load artifact + feature contract.
    artifact_dir = request.app.state.settings.artifact_dir
    try:
        payload = read_artifact(model["artifact_sha256"], artifact_dir)
    except FileNotFoundError as exc:
        raise InferenceError(
            status_code=502,
            error_code="artifact_file_missing",
            message=(
                f"artifact sha={model['artifact_sha256']} registered but not "
                f"on disk at {artifact_dir}. Re-sync the artifact."
            ),
            retryable=False,
            details={"sha256": model["artifact_sha256"]},
        ) from exc
    except ValueError as exc:  # sha mismatch / tamper
        raise InferenceError(
            status_code=502,
            error_code="artifact_corrupt",
            message=str(exc),
            retryable=False,
        ) from exc

    feature_names: list[str] = list(payload["feature_names"])
    versions = await features_repo.fetch_feature_versions(conn, feature_names)
    missing_versions = [n for n in feature_names if n not in versions]
    if missing_versions:
        raise InferenceError(
            status_code=409,
            error_code="feature_not_registered",
            message=(
                f"{len(missing_versions)} feature(s) referenced by the model "
                "are not in feature_registry (status=registered): "
                f"{missing_versions[:10]}"
            ),
            retryable=False,
            details={"missing": missing_versions},
        )

    feature_vector = await features_repo.fetch_per_bar_values_at_ts(
        conn,
        feature_names=feature_names,
        versions=versions,
        symbol=body.symbol,
        interval_name=body.interval_name,
        ts=body.ts,
    )

    try:
        value, confidence = predict_single(payload, feature_vector)
    except MissingFeatureError as exc:
        # Don't UPSERT a row — the JVM consumer's fail-open contract
        # already handles signal absence; we must not fabricate one.
        raise InferenceError(
            status_code=409,
            error_code="feature_value_missing",
            message=str(exc),
            retryable=True,
            hint=(
                "Run POST /features/{name}/v/{version}/backfill via the "
                "orchestrator to populate the missing rows, then retry."
            ),
            details={"missing": exc.missing},
        ) from exc
    except EmptyModelError as exc:
        # Artifact passed sha verification but its model body is absent.
        # blackheart-train producer bug — re-train and re-register.
        raise InferenceError(
            status_code=502,
            error_code="artifact_corrupt",
            message=str(exc),
            retryable=False,
        ) from exc

    meta = {
        "model_id": str(model["id"]),
        "artifact_sha256": model["artifact_sha256"],
        "spec_name": payload.get("spec", {}).get("name"),
        "objective": payload.get("objective"),
        "model_status_at_inference": model["status"],
    }
    await signals_repo.upsert_signal_value(
        conn,
        signal_id=body.signal_id,
        symbol=body.symbol,
        ts=body.ts,
        value=value,
        confidence=confidence,
        source=body.source,
        meta=meta,
        created_by=agent,
    )

    return {
        "signal_id": str(body.signal_id),
        "symbol": body.symbol,
        "ts": body.ts.isoformat(),
        "value": value,
        "confidence": confidence,
        "source": body.source,
        "model_id": str(model["id"]),
        "artifact_sha256": model["artifact_sha256"],
        "spec_name": payload.get("spec", {}).get("name"),
        "objective": payload.get("objective"),
    }


@router.post("/backfill")
async def run_backfill(
    body: BackfillBody,
    request: Request,
    conn: asyncpg.Connection = Depends(get_db_conn),
    agent: str = Depends(get_agent_name),
) -> dict[str, Any]:
    _validate_interval(body.interval_name)
    if body.end <= body.start:
        raise InferenceError(
            status_code=400,
            error_code="bad_window",
            message=f"end={body.end.isoformat()} must be strictly after start={body.start.isoformat()}.",
            retryable=False,
        )

    # Pre-flight cap from window width + interval — refuse oversized
    # windows BEFORE the heavy feature-fetch query, so a 10-year backfill
    # can't OOM the service on the SQL load. The post-fetch check below
    # is still there as a safety net for sparse-grid weirdness.
    estimated_bars = _estimate_backfill_bars(body.start, body.end, body.interval_name)
    max_rows = request.app.state.settings.max_backfill_rows
    if estimated_bars > max_rows:
        raise InferenceError(
            status_code=413,
            error_code="backfill_window_too_large",
            message=(
                f"window covers ~{estimated_bars} {body.interval_name} bars; "
                f"max_backfill_rows={max_rows}. Split the window into smaller calls."
            ),
            retryable=False,
            details={"estimated_bars": estimated_bars, "max_rows": max_rows},
        )

    signal, model = await _resolve_artifact(conn, body.signal_id)
    artifact_dir = request.app.state.settings.artifact_dir
    try:
        payload = read_artifact(model["artifact_sha256"], artifact_dir)
    except FileNotFoundError as exc:
        raise InferenceError(
            status_code=502,
            error_code="artifact_file_missing",
            message=str(exc),
            retryable=False,
        ) from exc

    feature_names: list[str] = list(payload["feature_names"])
    versions = await features_repo.fetch_feature_versions(conn, feature_names)
    missing_versions = [n for n in feature_names if n not in versions]
    if missing_versions:
        raise InferenceError(
            status_code=409,
            error_code="feature_not_registered",
            message=f"{len(missing_versions)} feature(s) not in feature_registry.",
            retryable=False,
            details={"missing": missing_versions},
        )

    grid = await features_repo.fetch_per_bar_values_range(
        conn,
        feature_names=feature_names,
        versions=versions,
        symbol=body.symbol,
        interval_name=body.interval_name,
        start=body.start,
        end=body.end,
    )

    max_rows = request.app.state.settings.max_backfill_rows
    if len(grid) > max_rows:
        raise InferenceError(
            status_code=413,
            error_code="backfill_window_too_large",
            message=(
                f"window covers {len(grid)} timestamps; max_backfill_rows={max_rows}. "
                "Split the window into smaller calls."
            ),
            retryable=False,
        )

    rows: list[tuple[datetime, dict[str, float | None]]] = sorted(
        ((ts, vec) for ts, vec in grid.items()), key=lambda kv: kv[0]
    )
    preds = predict_batch(payload, rows)

    to_write: list[dict[str, Any]] = []
    skipped = 0
    for ts, val in preds:
        if val is None:
            skipped += 1
            continue
        to_write.append({"ts": ts, "value": val, "confidence": None,
                         "meta": {
                             "model_id": str(model["id"]),
                             "artifact_sha256": model["artifact_sha256"],
                             "spec_name": payload.get("spec", {}).get("name"),
                             "objective": payload.get("objective"),
                         }})

    rows_written = 0
    if to_write:
        rows_written = await signals_repo.upsert_signal_values_batch(
            conn,
            signal_id=body.signal_id,
            symbol=body.symbol,
            rows=to_write,
            source=body.source,
            created_by=agent,
        )

    return {
        "signal_id": str(body.signal_id),
        "symbol": body.symbol,
        "interval_name": body.interval_name,
        "start": body.start.isoformat(),
        "end": body.end.isoformat(),
        "rows_in_window": len(grid),
        "rows_written": rows_written,
        "rows_skipped_missing_features": skipped,
        "model_id": str(model["id"]),
        "artifact_sha256": model["artifact_sha256"],
        "spec_name": payload.get("spec", {}).get("name"),
        "source": body.source,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
