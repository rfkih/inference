"""FastAPI app factory + lifespan."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from functools import partial

from fastapi import FastAPI

from .api.batch import router as batch_router
from .api.health import router as health_router
from .api.inference import router as inference_router
from .api.streaming import router as streaming_router
from .auth import AuthMiddleware
from .errors import register_exception_handlers
from .infra.db import Database
from .logging import configure_logging, get_logger
from .services.streaming import StreamingState, run_streaming_loop
from .settings import Settings

log = get_logger(__name__)


@asynccontextmanager
async def _lifespan(settings: Settings, app: FastAPI):
    # Fail fast at startup if a prod boot still carries the dev sentinel.
    # __main__ calls this too, but uvicorn-direct boot bypasses __main__,
    # so the lifespan is the single guaranteed enforcement point.
    settings.assert_prod_safe()
    db = Database(
        dsn=settings.db_dsn.get_secret_value(),
        min_size=settings.db_min_pool,
        max_size=settings.db_max_pool,
    )
    await db.open()
    app.state.settings = settings
    app.state.db = db
    app.state.streaming = StreamingState(
        enabled=settings.streaming_enabled,
        poll_seconds=settings.streaming_poll_seconds,
        max_gap_hours=settings.streaming_max_gap_hours,
    )
    streaming_task: asyncio.Task | None = None
    if settings.streaming_enabled:
        streaming_task = asyncio.create_task(
            run_streaming_loop(settings, db, app.state.streaming),
            name="inference-streaming",
        )
    log.info("inference.startup",
             host=settings.host, port=settings.port, profile=settings.profile,
             streaming=settings.streaming_enabled)
    try:
        yield
    finally:
        if streaming_task is not None:
            streaming_task.cancel()
            try:
                await streaming_task
            except (asyncio.CancelledError, BaseException):  # noqa: BLE001
                pass
        await db.close()
        log.info("inference.shutdown")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()  # type: ignore[call-arg]
    configure_logging(settings)

    app = FastAPI(
        title="Blackheart inference sidecar",
        version="0.1.0",
        description=(
            "Loads model_registry artifacts trained by blackheart-train, "
            "runs predictions, writes signal_history rows the trading "
            "JVM consumes."
        ),
        lifespan=partial(_lifespan, settings),
    )

    register_exception_handlers(app)
    app.add_middleware(AuthMiddleware, settings=settings)

    app.include_router(health_router)
    app.include_router(inference_router)
    app.include_router(streaming_router)
    app.include_router(batch_router)
    return app


# Module-level app for ``uvicorn blackheart_inference.main:app`` invocations.
app = create_app()
