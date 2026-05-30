"""Pydantic settings — env-prefixed ``INFERENCE_*``.

Matches the orchestrator's settings shape so operators can swap one for
the other in muscle memory: same auth-token guard, same loopback default,
same prod-profile self-check.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Dev sentinel — refused under profile='prod'. Matches the orchestrator's
# pattern so the same operational mistake (deploying prod with dev creds)
# is caught the same way across services.
_DEV_TOKEN_SENTINEL = "dev-token"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="INFERENCE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    profile: Literal["dev", "prod"] = "dev"

    auth_token: SecretStr = SecretStr(_DEV_TOKEN_SENTINEL)
    db_dsn: SecretStr = SecretStr(
        "postgresql://blackheart_research:CHANGEME@127.0.0.1:5432/blackheart"
    )

    # Where blackheart-train writes content-addressed .pkl artifacts.
    # Inference reads from the same path; the sha256 is the index.
    artifact_dir: Path = Path("C:/Project/blackheart-train/artifacts")

    host: str = "127.0.0.1"
    port: int = 8000

    # asyncpg pool sizing. Inference is light on DB — one connection per
    # active request is plenty for a single-worker uvicorn.
    db_min_pool: int = 1
    db_max_pool: int = 8

    # Feature-fetch tuning. The backfill endpoint queries a range and can
    # blow up if the range is large — cap defensively.
    max_backfill_rows: int = 50_000

    # ── live streaming worker ─────────────────────────────────────────────
    # When enabled, the sidecar runs a background loop that finds gaps
    # between MAX(signal_history.ts) and the latest available feature_values
    # row for every signal-symbol-interval tuple bound to a live ML gate,
    # and calls its own /inference/backfill to close them. Default OFF so
    # adding the capability doesn't surprise existing deployments — flip
    # via ``INFERENCE_STREAMING_ENABLED=true``.
    streaming_enabled: bool = False
    # How often the loop wakes up. 60s is enough granularity for 1h bars
    # (the smallest interval ML models currently use); a 5m model would
    # want this lower. Bounded below by 10s so a typo can't DoS the DB.
    streaming_poll_seconds: int = 60
    # Maximum gap the loop will attempt to backfill in one tick. Older
    # gaps require a manual /inference/backfill call (operator decides
    # whether the stale window is worth replaying). Prevents a stalled
    # signal from spawning a multi-day backfill on resume.
    streaming_max_gap_hours: int = 24

    def assert_prod_safe(self) -> None:
        """Refuse to boot under prod with insecure defaults.

        Same contract the orchestrator's ``Settings.assert_prod_safe``
        uses — fails fast at startup rather than letting the service
        come up with the dev sentinel still in place.
        """
        if self.profile != "prod":
            return
        if self.auth_token.get_secret_value() == _DEV_TOKEN_SENTINEL:
            raise RuntimeError(
                "INFERENCE_AUTH_TOKEN is the dev sentinel under profile=prod"
            )
        if self.host not in ("127.0.0.1", "0.0.0.0"):
            raise RuntimeError(
                f"INFERENCE_HOST={self.host!r} under profile=prod; "
                "must be 127.0.0.1 (bare-metal) or 0.0.0.0 (Docker — port isolation via compose)"
            )
