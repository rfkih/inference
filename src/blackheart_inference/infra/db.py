"""asyncpg pool + JSONB codec.

JSONB columns (model_registry.metrics, signal_history.meta, ...) are
dict-on-the-wire — same convention as the orchestrator. Pass dicts
directly to ``$N`` params; do NOT ``json.dumps`` first.
"""

from __future__ import annotations

import json
from typing import Any

import asyncpg


def _json_dumps(value: Any) -> str:
    return json.dumps(value, default=str)


async def _init_connection(conn: asyncpg.Connection) -> None:
    # Mirror the orchestrator's codec so JSONB roundtrips as dict.
    await conn.set_type_codec(
        "jsonb",
        encoder=_json_dumps,
        decoder=json.loads,
        schema="pg_catalog",
    )
    await conn.set_type_codec(
        "json",
        encoder=_json_dumps,
        decoder=json.loads,
        schema="pg_catalog",
    )


class Database:
    """Thin asyncpg pool wrapper. Mirrors the orchestrator's shape so the
    two services have one ops surface."""

    def __init__(self, dsn: str, *, min_size: int, max_size: int) -> None:
        self._dsn = dsn
        self._min_size = min_size
        self._max_size = max_size
        self._pool: asyncpg.Pool | None = None

    async def open(self) -> None:
        if self._pool is not None:
            return
        self._pool = await asyncpg.create_pool(
            dsn=self._dsn,
            min_size=self._min_size,
            max_size=self._max_size,
            init=_init_connection,
        )

    async def close(self) -> None:
        if self._pool is None:
            return
        await self._pool.close()
        self._pool = None

    def acquire(self):
        if self._pool is None:
            raise RuntimeError("Database.open() not called before acquire().")
        return self._pool.acquire()

    async def health_probe(self) -> bool:
        if self._pool is None:
            return False
        try:
            async with self.acquire() as conn:
                v = await conn.fetchval("SELECT 1")
                return v == 1
        except Exception:
            return False
