"""Unit tests for the live-streaming worker's gap-detection logic.

Pure-function tests of :func:`find_gap_for_target` via an in-memory fake
conn — no DB, no httpx. The full loop is exercised in the smoke-test
suite against the running sidecar.

The reason-code semantics this test pins:
* ``up_to_date``     — no gap; do nothing this tick.
* ``no_features``    — no feature_values rows yet; can't predict.
* ``gap_too_large``  — stale beyond ``streaming_max_gap_hours``; refuse;
                       operator must catch up manually.
* ``ok``             — gap exists and is within budget; backfill it.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest

from blackheart_inference.services.streaming import (
    _INTERVAL_SECONDS,
    StreamingTarget,
    find_gap_for_target,
)


_SIGNAL_ID = UUID("53141c26-76c6-46e8-b503-cc7612eb23fa")


def _target(symbol: str = "ETHUSDT", interval: str = "1h") -> StreamingTarget:
    return StreamingTarget(
        signal_id=_SIGNAL_ID,
        signal_name="regime_eth_v2",
        status="active",
        symbol=symbol,
        interval_name=interval,
    )


class _FakeConn:
    """Minimal asyncpg-shaped fake. ``fetchrow`` returns a single
    dict that the production query would have returned."""
    def __init__(self, *, last_signal_ts: datetime | None, latest_feature_ts: datetime | None):
        self._row = {
            "last_signal_ts": last_signal_ts,
            "latest_feature_ts": latest_feature_ts,
        }

    async def fetchrow(self, _sql: str, *_args):
        return self._row


@pytest.mark.asyncio
async def test_no_features_returns_none() -> None:
    """A signal with no feature_values rows yet must surface a clear
    'no_features' reason — the worker can't fabricate predictions."""
    conn = _FakeConn(last_signal_ts=None, latest_feature_ts=None)
    gap, reason = await find_gap_for_target(conn, _target(), max_gap_hours=24)
    assert gap is None
    assert reason == "no_features"


@pytest.mark.asyncio
async def test_fresh_signal_clamps_to_max_gap_hours() -> None:
    """Brand-new signal (no signal_history) should start ``max_gap_hours``
    before the latest feature, not from the dawn of time."""
    latest = datetime(2026, 5, 23, 12, 0, tzinfo=timezone.utc)
    conn = _FakeConn(last_signal_ts=None, latest_feature_ts=latest)
    gap, reason = await find_gap_for_target(conn, _target(), max_gap_hours=24)
    assert reason == "ok"
    assert gap is not None
    # End is exclusive — one second past the latest available bar.
    assert gap.end == latest + timedelta(seconds=1)
    # Fresh-signal clamp is anchored on ``end`` (not ``latest``) so the
    # span budget treats the boundary case as ``ok``, not ``gap_too_large``.
    assert gap.start == gap.end - timedelta(hours=24)


@pytest.mark.asyncio
async def test_up_to_date_returns_none_when_signal_matches_latest() -> None:
    """Last signal exactly equals latest feature → next bar hasn't been
    published yet → nothing to do."""
    latest = datetime(2026, 5, 23, 12, 0, tzinfo=timezone.utc)
    conn = _FakeConn(last_signal_ts=latest, latest_feature_ts=latest)
    gap, reason = await find_gap_for_target(conn, _target(), max_gap_hours=24)
    # last+1h > latest+1s, so end <= start, so up_to_date.
    assert gap is None
    assert reason == "up_to_date"


@pytest.mark.asyncio
async def test_short_gap_within_budget_returns_ok() -> None:
    """1h cadence, 3-hour stall — well within 24h budget."""
    latest = datetime(2026, 5, 23, 12, 0, tzinfo=timezone.utc)
    last_signal = latest - timedelta(hours=3)
    conn = _FakeConn(last_signal_ts=last_signal, latest_feature_ts=latest)
    gap, reason = await find_gap_for_target(conn, _target(), max_gap_hours=24)
    assert reason == "ok"
    assert gap is not None
    # First bar to predict is last_signal + 1h
    assert gap.start == last_signal + timedelta(hours=1)
    assert gap.end == latest + timedelta(seconds=1)


@pytest.mark.asyncio
async def test_gap_exceeding_budget_refuses() -> None:
    """24h budget, signal stale for 48h → refuse. Operator must catch
    up via manual /inference/backfill."""
    latest = datetime(2026, 5, 23, 12, 0, tzinfo=timezone.utc)
    last_signal = latest - timedelta(hours=48)
    conn = _FakeConn(last_signal_ts=last_signal, latest_feature_ts=latest)
    gap, reason = await find_gap_for_target(conn, _target(), max_gap_hours=24)
    assert gap is None
    assert reason == "gap_too_large"


@pytest.mark.asyncio
async def test_4h_interval_works() -> None:
    """Sanity check on intervals other than 1h — the worker uses the
    target's interval_name to compute the next-bar offset."""
    latest = datetime(2026, 5, 23, 12, 0, tzinfo=timezone.utc)
    last_signal = latest - timedelta(hours=8)
    conn = _FakeConn(last_signal_ts=last_signal, latest_feature_ts=latest)
    gap, reason = await find_gap_for_target(conn, _target(interval="4h"), max_gap_hours=24)
    assert reason == "ok"
    assert gap is not None
    # 4h interval — next bar is last + 4h
    assert gap.start == last_signal + timedelta(hours=4)


def test_interval_seconds_map_pinned() -> None:
    """Mirror of the values in api/inference.py. A future divergence
    must be loud — a 1h interval treated as 60s in the worker would
    miss bars; a 5m interval treated as 3600s would over-predict."""
    assert _INTERVAL_SECONDS == {
        "5m":   300,
        "15m":  900,
        "1h":   3_600,
        "4h":   14_400,
    }
