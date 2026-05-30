"""Shared fixtures.

Tests build the app with explicit ``Settings`` (no .env required) and
override the DB / auth surfaces as needed. DB-touching paths are not
exercised here — those need pytest-postgresql + real migrations and
belong in an integration suite.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from blackheart_inference.main import create_app
from blackheart_inference.settings import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return Settings(
        profile="dev",
        auth_token=SecretStr("test-token"),
        db_dsn=SecretStr("postgresql://x:y@127.0.0.1:5432/none"),
        artifact_dir=artifact_dir,
    )


@pytest.fixture
def client(settings: Settings) -> TestClient:
    app = create_app(settings)
    return TestClient(app, raise_server_exceptions=False)
