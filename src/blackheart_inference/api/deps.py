"""FastAPI dependencies."""

from __future__ import annotations

from collections.abc import AsyncIterator

import asyncpg
from fastapi import Request


async def get_db_conn(request: Request) -> AsyncIterator[asyncpg.Connection]:
    async with request.app.state.db.acquire() as conn:
        yield conn


def get_agent_name(request: Request) -> str:
    return getattr(request.state, "agent_name", "anonymous")
