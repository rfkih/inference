"""``GET /streaming/status`` — surface the live-streaming worker's state.

Read-only, idempotent, no DB. Just reflects the in-process ``StreamingState``
attached to ``app.state.streaming`` by the lifespan hook. The trading-JVM
dashboard (Phase 1 frontend ``MlHealthStrip``) can hit this to verify
the worker is alive without parsing logs.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Request


router = APIRouter(prefix="/streaming", tags=["streaming"])


def _iter_to_dict(it: Any) -> dict[str, Any] | None:
    if it is None:
        return None
    return {
        "startedAt": it.started_at.isoformat() if it.started_at else None,
        "endedAt": it.ended_at.isoformat() if it.ended_at else None,
        "targetsTotal": it.targets_total,
        "targetsUpToDate": it.targets_up_to_date,
        "targetsNoFeatures": it.targets_no_features,
        "targetsGapTooLarge": it.targets_gap_too_large,
        "targetsBackfilled": it.targets_backfilled,
        "rowsWrittenTotal": it.rows_written_total,
        "errors": list(it.errors)[:10],   # cap so a flood of errors doesn't bloat the response
    }


@router.get("/status")
async def get_streaming_status(request: Request) -> dict[str, Any]:
    state = getattr(request.app.state, "streaming", None)
    if state is None:
        return {
            "enabled": False,
            "reason": "streaming worker not configured on this process",
        }
    return {
        "enabled": state.enabled,
        "pollSeconds": state.poll_seconds,
        "maxGapHours": state.max_gap_hours,
        "startedAt": state.started_at.isoformat() if state.started_at else None,
        "iterations": state.iterations,
        "lastIteration": _iter_to_dict(state.last_iteration),
        "lastErrorAt": state.last_error_at.isoformat() if state.last_error_at else None,
        "lastError": state.last_error,
        "now": datetime.utcnow().isoformat(),
    }
