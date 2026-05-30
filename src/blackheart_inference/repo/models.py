"""Read ``model_registry`` + ``signal_definition``.

The inference flow needs to resolve ``signal_id → model_id → artifact_sha256``
before it can load the booster. Both tables are owned by the trading
JVM's V66 migration; we only read them.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

import asyncpg


async def fetch_signal_definition(
    conn: asyncpg.Connection, signal_id: UUID
) -> dict[str, Any] | None:
    """Return ``{signal_id, name, model_id, status, value_range, ...}`` or None.

    Inference accepts shadow/active signals; refuses retired ones — a
    retired signal shouldn't be written to ``signal_history`` because
    the JVM consumer wouldn't gate on it anyway.
    """
    row = await conn.fetchrow(
        """
        SELECT signal_id, name, model_id, horizon, value_range,
               description, status
          FROM signal_definition
         WHERE signal_id = $1
        """,
        signal_id,
    )
    return dict(row) if row else None


async def fetch_model_registry_row(
    conn: asyncpg.Connection, model_id: UUID
) -> dict[str, Any] | None:
    """Return the full model_registry row, including artifact_sha256 + status.

    Inference refuses to load:
      * ``status='retired'`` rows (the artifact may still exist on disk
        but promoting it would defeat the lifecycle)
      * ``status='rejected_by_operator'`` rows (same reason)
      * rows missing ``artifact_sha256`` (incomplete registration)

    The orchestrator's promotion gate keys on status — we don't
    second-guess what's appropriate to serve, only refuse the obvious.
    """
    row = await conn.fetchrow(
        """
        SELECT id, family, purpose, symbol, interval, horizon_bars,
               serving_interval, feature_set, training_window,
               hyperparams, metrics, random_seed, code_commit,
               artifact_uri, artifact_sha256, artifact_size_bytes,
               artifact_synced_to_vps, status, version
          FROM model_registry
         WHERE id = $1
        """,
        model_id,
    )
    return dict(row) if row else None
