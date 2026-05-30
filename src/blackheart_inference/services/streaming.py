"""Live-streaming worker — closes the gap between ``signal_history`` and
the latest available ``feature_values`` row for every signal bound to a
live ML gate.

Bridges the README-documented "Not built" item:

    | Live-streaming worker (cron-fires inference on every new bar) | …

Design (2026-05-23):

* The loop runs as an asyncio task started from the sidecar's lifespan
  hook when ``INFERENCE_STREAMING_ENABLED=true``. Single-process by
  construction; same Booster cache as the on-demand /inference path.
* Per tick: find every ``(signal, symbol, interval_name)`` tuple where
  ``account_strategy.ml_regime_signal_name = signal.name`` AND
  ``ml_regime_gate_enabled = TRUE`` AND ``signal.status IN ('active',
  'shadow')``. Empty when no operator has wired any gate — by design
  the worker spends zero CPU on signals nobody is using.
* For each tuple, compute the gap between MAX(signal_history.ts) and
  MAX(feature_values.ts) for the (symbol, interval). If non-empty AND
  within ``streaming_max_gap_hours``, call the local ``/inference/
  backfill`` via loopback httpx (re-uses the existing predict + UPSERT
  path; no duplication of business logic).
* Errors on one tuple do NOT halt the loop — the failing tuple is
  logged and the next tuple is processed. The loop only exits on
  cancellation (lifespan shutdown).

Status is exposed via ``GET /streaming/status`` (api/streaming.py) so
the operator dashboard can verify the worker is alive without grep-ing
journald.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

import httpx

from ..infra.db import Database
from ..logging import get_logger
from ..settings import Settings


log = get_logger(__name__)


# Interval → seconds. Mirrored from api/inference.py — kept inline so a
# future divergence is loud rather than silent (a unit test pins these).
_INTERVAL_SECONDS: dict[str, int] = {"5m": 300, "15m": 900, "1h": 3_600, "4h": 14_400}


@dataclass(frozen=True)
class StreamingTarget:
    """One signal-symbol-interval combo the worker should keep current."""
    signal_id: UUID
    signal_name: str
    status: str               # 'active' | 'shadow'
    symbol: str
    interval_name: str


@dataclass(frozen=True)
class Gap:
    """Half-open ``[start, end)`` window to backfill."""
    start: datetime
    end: datetime


@dataclass
class IterationResult:
    """Summary of one tick. Surfaced by ``GET /streaming/status``."""
    started_at: datetime
    ended_at: datetime | None = None
    targets_total: int = 0
    targets_up_to_date: int = 0
    targets_no_features: int = 0
    targets_gap_too_large: int = 0
    targets_backfilled: int = 0
    rows_written_total: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass
class StreamingState:
    """Process-wide rolling status — single instance lives on
    ``app.state.streaming`` for the status endpoint."""
    enabled: bool
    poll_seconds: int
    max_gap_hours: int
    started_at: datetime | None = None
    iterations: int = 0
    last_iteration: IterationResult | None = None
    last_error_at: datetime | None = None
    last_error: str | None = None


_TARGETS_SQL = """
SELECT DISTINCT
    sd.signal_id,
    sd.name           AS signal_name,
    sd.status         AS signal_status,
    mr.symbol         AS symbol,
    mr.interval       AS interval_name
FROM signal_definition sd
JOIN model_registry mr ON mr.id = sd.model_id
WHERE sd.status IN ('active', 'shadow')
  AND mr.symbol IS NOT NULL
  AND mr.interval IS NOT NULL
  AND mr.interval = ANY($1::text[])
ORDER BY sd.name, mr.symbol, mr.interval
"""


async def find_streaming_targets(conn) -> list[StreamingTarget]:
    """Query all active/shadow signals and keep them fresh via their
    model's symbol+interval. Previously gated on ml_regime_gate_enabled=TRUE
    which left signal_history empty whenever no live gate was enabled,
    making the ML monitor always show 0% coverage."""
    intervals = list(_INTERVAL_SECONDS.keys())
    records = await conn.fetch(_TARGETS_SQL, intervals)
    return [
        StreamingTarget(
            signal_id=r["signal_id"],
            signal_name=r["signal_name"],
            status=r["signal_status"],
            symbol=r["symbol"],
            interval_name=r["interval_name"],
        )
        for r in records
    ]


_GAP_SQL = """
SELECT
    (
        SELECT MAX(ts)
        FROM signal_history
        WHERE signal_id = $1 AND symbol = $2
    ) AS last_signal_ts,
    (
        SELECT MAX(ts)
        FROM feature_values
        WHERE symbol = $2 AND interval = $3
    ) AS latest_feature_ts
"""


async def find_gap_for_target(
    conn,
    target: StreamingTarget,
    max_gap_hours: int,
) -> tuple[Gap | None, str]:
    """Return ``(gap, reason_code)``. ``gap=None`` means "do nothing this
    tick"; the reason code is one of ``up_to_date``, ``no_features``,
    ``gap_too_large``, ``ok``.

    Gap semantics:
    * start = last_signal_ts + interval (the next bar after the most
      recent persisted one). When no signal history exists yet, start
      from ``latest_feature_ts - max_gap_hours``.
    * end = latest_feature_ts + 1s (so the backfill endpoint's exclusive
      end includes the latest available bar).
    * If the resulting window exceeds ``max_gap_hours``, the worker
      refuses — operator must manually catch up via /inference/backfill,
      so a long-stalled signal doesn't spawn an enormous loop tick on
      resume.
    """
    row = await conn.fetchrow(_GAP_SQL, target.signal_id, target.symbol, target.interval_name)
    latest_feat_ts: datetime | None = row["latest_feature_ts"]
    last_signal_ts: datetime | None = row["last_signal_ts"]

    if latest_feat_ts is None:
        return None, "no_features"

    interval_seconds = _INTERVAL_SECONDS[target.interval_name]
    # Exclusive end — include the latest available bar.
    end = latest_feat_ts + timedelta(seconds=1)
    if last_signal_ts is None:
        # Fresh signal — clamp the window to exactly ``max_gap_hours``
        # measured from ``end`` (not from ``latest_feat_ts``) so the
        # span-budget check below treats the boundary case as "ok"
        # rather than ``gap_too_large`` (off-by-1s otherwise).
        start = end - timedelta(hours=max_gap_hours)
    else:
        start = last_signal_ts + timedelta(seconds=interval_seconds)

    if end <= start:
        return None, "up_to_date"

    span_hours = (end - start).total_seconds() / 3_600.0
    if span_hours > max_gap_hours:
        return None, "gap_too_large"

    return Gap(start=start, end=end), "ok"


async def call_backfill(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    auth_token: str,
    target: StreamingTarget,
    gap: Gap,
) -> dict[str, Any]:
    """Loopback call to the sidecar's own /inference/backfill. Reuses
    the existing predict + UPSERT path — no duplication of business
    logic, single source of truth for feature-fetch + skip semantics."""
    body = {
        "signal_id": str(target.signal_id),
        "symbol": target.symbol,
        "interval_name": target.interval_name,
        "start": gap.start.isoformat(),
        "end": gap.end.isoformat(),
        # 'catchup_scan' = the worker is filling a small gap on a fresh
        # signal or after a brief stall. The trading-JVM consumer treats
        # all three sources equally; this just makes journal queries
        # legible later ("where did this row come from?").
        "source": "catchup_scan",
    }
    response = await client.post(
        f"{base_url}/inference/backfill",
        json=body,
        headers={"X-Inference-Token": auth_token, "X-Agent-Name": "streaming-worker"},
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()


async def streaming_iteration(
    settings: Settings,
    db: Database,
    client: httpx.AsyncClient,
) -> IterationResult:
    """One pass over all targets. Returns a structured summary even when
    individual targets failed — caller decides whether to log loud."""
    result = IterationResult(started_at=datetime.now(timezone.utc))
    try:
        async with db.acquire() as conn:
            targets = await find_streaming_targets(conn)
    except Exception as exc:  # noqa: BLE001 — DB-level error; surface to caller
        result.errors.append(f"find_targets: {exc!r}")
        result.ended_at = datetime.now(timezone.utc)
        return result

    result.targets_total = len(targets)
    base_url = f"http://{settings.host}:{settings.port}"
    auth_token = settings.auth_token.get_secret_value()

    for target in targets:
        try:
            async with db.acquire() as conn:
                gap, reason = await find_gap_for_target(
                    conn, target, max_gap_hours=settings.streaming_max_gap_hours,
                )
        except Exception as exc:  # noqa: BLE001
            result.errors.append(
                f"gap[{target.signal_name}/{target.symbol}/{target.interval_name}]: {exc!r}"
            )
            continue

        if reason == "up_to_date":
            result.targets_up_to_date += 1
            continue
        if reason == "no_features":
            result.targets_no_features += 1
            continue
        if reason == "gap_too_large":
            result.targets_gap_too_large += 1
            log.warning(
                "streaming.gap_too_large",
                signal=target.signal_name, symbol=target.symbol,
                interval=target.interval_name,
                max_gap_hours=settings.streaming_max_gap_hours,
            )
            continue

        assert gap is not None  # reason == "ok"
        try:
            payload = await call_backfill(
                client, base_url=base_url, auth_token=auth_token,
                target=target, gap=gap,
            )
        except httpx.HTTPStatusError as exc:
            body = ""
            try:
                body = exc.response.text[:300]
            except Exception:  # noqa: BLE001 — best-effort log only
                pass
            result.errors.append(
                f"backfill[{target.signal_name}/{target.symbol}]: "
                f"HTTP {exc.response.status_code} {body}"
            )
            log.warning(
                "streaming.backfill_http_error",
                signal=target.signal_name, symbol=target.symbol,
                status=exc.response.status_code,
            )
            continue
        except Exception as exc:  # noqa: BLE001
            result.errors.append(
                f"backfill[{target.signal_name}/{target.symbol}]: {exc!r}"
            )
            log.warning(
                "streaming.backfill_error",
                signal=target.signal_name, symbol=target.symbol, error=repr(exc),
            )
            continue

        rows_written = int(payload.get("rows_written", 0) or 0)
        result.targets_backfilled += 1
        result.rows_written_total += rows_written
        if rows_written > 0:
            log.info(
                "streaming.backfill_ok",
                signal=target.signal_name, symbol=target.symbol,
                interval=target.interval_name,
                rows_written=rows_written,
                start=gap.start.isoformat(), end=gap.end.isoformat(),
            )

    result.ended_at = datetime.now(timezone.utc)
    return result


async def run_streaming_loop(
    settings: Settings,
    db: Database,
    state: StreamingState,
) -> None:
    """Long-running loop. Started from the sidecar's lifespan when
    ``streaming_enabled=true``. Sleeps ``streaming_poll_seconds``
    between ticks; cancellable via task.cancel() at shutdown.

    Per-iteration errors land in ``state.last_error`` but never raise
    out of the loop — the worker is a "best effort, keep going" service.
    """
    state.started_at = datetime.now(timezone.utc)
    log.info(
        "streaming.start",
        poll_seconds=settings.streaming_poll_seconds,
        max_gap_hours=settings.streaming_max_gap_hours,
    )
    # The HTTP client is per-loop, not per-iteration — keep the keepalive
    # connection warm so the loopback handshake doesn't recur every tick.
    async with httpx.AsyncClient() as client:
        try:
            while True:
                try:
                    result = await streaming_iteration(settings, db, client)
                    state.iterations += 1
                    state.last_iteration = result
                    if result.errors:
                        state.last_error_at = datetime.now(timezone.utc)
                        state.last_error = "; ".join(result.errors[:3])
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    # Catch-all so an unexpected error doesn't kill the
                    # loop. Logged loud; state.last_error surfaces to the
                    # status endpoint for operator visibility.
                    log.exception("streaming.iteration_unhandled", error=repr(exc))
                    state.last_error_at = datetime.now(timezone.utc)
                    state.last_error = f"unhandled: {exc!r}"
                await asyncio.sleep(max(10, settings.streaming_poll_seconds))
        except asyncio.CancelledError:
            log.info("streaming.cancelled")
            raise
