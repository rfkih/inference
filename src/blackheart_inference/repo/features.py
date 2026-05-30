"""Read ``feature_values`` + ``feature_registry``.

Inference needs:
  * the registered version of each feature in the model's feature_set
  * the values at the timestamp(s) being inferred

The schema (V70) keys feature_values on
``(feature_name, version, symbol, interval, ts)``; per-bar features
carry ``symbol`` and ``interval`` filled, global features carry empty
strings. We honor both via the spec's ``feature_set`` payload.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import asyncpg


async def get_latest_ts(
    conn: asyncpg.Connection,
) -> datetime | None:
    """Get the latest timestamp from feature_values.

    Returns None if no rows exist in feature_values (empty table).
    """
    row = await conn.fetchrow(
        """
        SELECT MAX(ts) as latest_ts
          FROM feature_values
        """
    )
    if row and row["latest_ts"]:
        return row["latest_ts"]
    return None


async def fetch_feature_versions(
    conn: asyncpg.Connection, feature_names: list[str]
) -> dict[str, int]:
    """Map feature_name → registered version.

    Inference uses the latest ``registered`` version for each feature —
    matches blackheart-train's loader contract. If a feature is missing
    from the registry, the caller will detect the gap when building the
    feature vector and fail with a clear error.

    Returns an empty dict for an empty input — no SQL fired.
    """
    if not feature_names:
        return {}
    rows = await conn.fetch(
        """
        SELECT feature_name, version
          FROM feature_registry
         WHERE feature_name = ANY($1::text[])
           AND status = 'registered'
        """,
        feature_names,
    )
    # If multiple versions exist (rare; a future bug in registry hygiene),
    # take the max. blackheart-train sorts by version in its training
    # loader; we match that.
    out: dict[str, int] = {}
    for r in rows:
        name = r["feature_name"]
        v = int(r["version"])
        if name not in out or v > out[name]:
            out[name] = v
    return out


async def fetch_per_bar_values_at_ts(
    conn: asyncpg.Connection,
    *,
    feature_names: list[str],
    versions: dict[str, int],
    symbol: str,
    interval_name: str,
    ts: datetime,
) -> dict[str, float | None]:
    """Return ``{feature_name: value or None}`` at exactly ``ts``.

    None entries surface as the caller's "missing feature" error —
    inference cannot fabricate a value. PIT-correctness: ``ts`` is
    inclusive, no forward-fill at the boundary. (Forward-fill across
    older bars is the registry-time / training-time concern; at
    inference we honour what's on the row.)
    """
    if not feature_names:
        return {}
    # Build per-row version filter via a CTE so a single query handles
    # the heterogeneous (name, version) pairs.
    pairs = [(name, versions.get(name)) for name in feature_names]
    names_arr = [p[0] for p in pairs]
    versions_arr: list[int] = [int(p[1]) if p[1] is not None else -1 for p in pairs]
    rows = await conn.fetch(
        """
        WITH wanted AS (
          SELECT unnest($1::text[]) AS feature_name,
                 unnest($2::int[])  AS version
        )
        SELECT fv.feature_name, fv.value
          FROM feature_values fv
          JOIN wanted w
            ON w.feature_name = fv.feature_name
           AND w.version      = fv.version
         WHERE fv.symbol   = $3
           AND fv.interval = $4
           AND fv.ts       = $5
        """,
        names_arr, versions_arr, symbol, interval_name, ts,
    )
    out: dict[str, float | None] = {name: None for name in feature_names}
    for r in rows:
        v = r["value"]
        out[r["feature_name"]] = float(v) if v is not None else None
    return out


async def fetch_per_bar_values_range(
    conn: asyncpg.Connection,
    *,
    feature_names: list[str],
    versions: dict[str, int],
    symbol: str,
    interval_name: str,
    start: datetime,
    end: datetime,
) -> dict[datetime, dict[str, float | None]]:
    """Return ``{ts: {feature_name: value or None}}`` over ``[start, end)``.

    One SQL query, post-pivoted in Python. Cap enforcement lives at the
    API layer (``Settings.max_backfill_rows``) — repo trusts inputs.
    """
    if not feature_names:
        return {}
    names_arr = [name for name in feature_names]
    versions_arr: list[int] = [int(versions.get(name, -1)) for name in feature_names]
    rows = await conn.fetch(
        """
        WITH wanted AS (
          SELECT unnest($1::text[]) AS feature_name,
                 unnest($2::int[])  AS version
        )
        SELECT fv.ts, fv.feature_name, fv.value
          FROM feature_values fv
          JOIN wanted w
            ON w.feature_name = fv.feature_name
           AND w.version      = fv.version
         WHERE fv.symbol   = $3
           AND fv.interval = $4
           AND fv.ts >= $5
           AND fv.ts <  $6
         ORDER BY fv.ts ASC
        """,
        names_arr, versions_arr, symbol, interval_name, start, end,
    )
    out: dict[datetime, dict[str, float | None]] = {}
    for r in rows:
        ts = r["ts"]
        slot = out.setdefault(ts, {name: None for name in feature_names})
        v = r["value"]
        slot[r["feature_name"]] = float(v) if v is not None else None
    return out
