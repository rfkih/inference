"""``/healthz`` (no I/O) and ``/readyz`` (DB probe)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from .. import __version__

router = APIRouter(tags=["health"])


@router.get("/healthz")
async def healthz() -> dict[str, Any]:
    return {"status": "ok", "version": __version__}


@router.get("/readyz")
async def readyz(request: Request) -> dict[str, Any]:
    db_ok = await request.app.state.db.health_probe()
    artifact_dir_exists = request.app.state.settings.artifact_dir.exists()
    return {
        "status": "ok" if (db_ok and artifact_dir_exists) else "degraded",
        "db": db_ok,
        "artifact_dir": artifact_dir_exists,
    }
