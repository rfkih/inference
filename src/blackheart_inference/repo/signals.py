"""Write ``signal_history``.

Per V66 the table is range-partitioned on ``ts`` with partitions
pre-created through 2028. The trading JVM's ``MLSignalService`` reads
the rows we write here. Schema enforces ``source IN ('stream',
'catchup_scan', 'historical_replay')`` — inference picks one based on
the caller's intent.

Upsert semantics: ``(signal_id, symbol, ts)`` is the PK. A repeat
inference (e.g. researcher re-runs a backfill after refining the model)
overwrites the prior value. We preserve the original ``created_time``
via DO UPDATE on value/produced_at/source/meta — keeps the audit field
honest about when the row first appeared, while latest content wins.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

import asyncpg


VALID_SOURCES = frozenset({"stream", "catchup_scan", "historical_replay"})


async def fetch_active_signals(
    conn: asyncpg.Connection,
) -> list[dict[str, Any]]:
    """Fetch all active or shadow signals ready for batch inference.

    Returns list of signal_definition rows (signal_id, name, model_id,
    status, horizon, value_range, description). Filters out retired and
    rejected_by_operator signals.
    """
    rows = await conn.fetch(
        """
        SELECT signal_id, name, model_id, horizon, value_range,
               description, status
          FROM signal_definition
         WHERE status IN ('active', 'shadow')
        """
    )
    return [dict(r) for r in rows]


async def upsert_signal_value(
    conn: asyncpg.Connection,
    *,
    signal_id: UUID,
    symbol: str,
    ts: datetime,
    value: float,
    confidence: float | None,
    source: str,
    meta: dict[str, Any] | None,
    created_by: str,
) -> None:
    if source not in VALID_SOURCES:
        raise ValueError(
            f"signal_history.source={source!r} not in {sorted(VALID_SOURCES)}"
        )
    await conn.execute(
        """
        INSERT INTO signal_history
          (signal_id, symbol, ts, value, confidence, produced_at,
           source, meta, created_by, updated_by)
        VALUES ($1, $2, $3, $4, $5, NOW(), $6, $7, $8, $8)
        ON CONFLICT (signal_id, symbol, ts) DO UPDATE
          SET value       = EXCLUDED.value,
              confidence  = EXCLUDED.confidence,
              produced_at = EXCLUDED.produced_at,
              source      = EXCLUDED.source,
              meta        = EXCLUDED.meta,
              updated_time = NOW(),
              updated_by   = EXCLUDED.updated_by
        """,
        signal_id, symbol, ts, value, confidence,
        source, meta or {}, created_by,
    )


async def upsert_signal_values_batch(
    conn: asyncpg.Connection,
    *,
    signal_id: UUID,
    symbol: str,
    rows: list[dict[str, Any]],
    source: str,
    created_by: str,
) -> int:
    """Batch UPSERT for backfill. ``rows`` items carry ``ts``, ``value``,
    optional ``confidence``, optional ``meta``. Returns the row count
    written.

    Uses one ``executemany``-style call so partition pruning still works
    (each row carries its own ``ts``).
    """
    if source not in VALID_SOURCES:
        raise ValueError(
            f"signal_history.source={source!r} not in {sorted(VALID_SOURCES)}"
        )
    if not rows:
        return 0
    args = [
        (
            signal_id, symbol, r["ts"], float(r["value"]),
            (None if r.get("confidence") is None else float(r["confidence"])),
            source, r.get("meta") or {}, created_by,
        )
        for r in rows
    ]
    await conn.executemany(
        """
        INSERT INTO signal_history
          (signal_id, symbol, ts, value, confidence, produced_at,
           source, meta, created_by, updated_by)
        VALUES ($1, $2, $3, $4, $5, NOW(), $6, $7, $8, $8)
        ON CONFLICT (signal_id, symbol, ts) DO UPDATE
          SET value       = EXCLUDED.value,
              confidence  = EXCLUDED.confidence,
              produced_at = EXCLUDED.produced_at,
              source      = EXCLUDED.source,
              meta        = EXCLUDED.meta,
              updated_time = NOW(),
              updated_by   = EXCLUDED.updated_by
        """,
        args,
    )
    return len(args)
