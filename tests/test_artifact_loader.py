"""Test artifact_loader against real blackheart-train artifacts when
available, plus a synthesised artifact for the round-trip / tamper cases.

Skipped if blackheart-train's artifact directory isn't on disk — keeps
the suite green on a fresh checkout without the training side mounted.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path

import pytest

from blackheart_inference.services.artifact_loader import (
    artifact_path,
    read_artifact,
)


_REAL_ARTIFACT_DIR = Path("C:/Project/blackheart-train/artifacts")


def _find_one_real_artifact_sha() -> str | None:
    """Locate any real artifact sha to exercise the loader against
    LightGBM-produced bytes. Returns None when no artifacts exist."""
    if not _REAL_ARTIFACT_DIR.exists():
        return None
    for shard in sorted(os.listdir(_REAL_ARTIFACT_DIR)):
        shard_dir = _REAL_ARTIFACT_DIR / shard
        if not shard_dir.is_dir():
            continue
        for f in os.listdir(shard_dir):
            if f.endswith(".pkl"):
                return f[:-4]
    return None


@pytest.mark.skipif(
    _find_one_real_artifact_sha() is None,
    reason="No real blackheart-train artifacts on disk; skipping integration check.",
)
def test_read_real_artifact_round_trips_sha() -> None:
    sha = _find_one_real_artifact_sha()
    assert sha is not None
    payload = read_artifact(sha, _REAL_ARTIFACT_DIR)
    assert payload["content_sha256"] == sha
    # blackheart-train v2 artifacts persist feature_names as a tuple
    # (ds.feature_names is built via tuple() in loader). Inference coerces
    # to list at the predictor boundary; this loader doesn't transform.
    assert isinstance(payload["feature_names"], (list, tuple))
    assert len(payload["feature_names"]) > 0
    assert payload["objective"] in ("binary", "regression", "multiclass")
    # v1 → v2 backfill: keys must be populated even on old artifacts.
    assert "payload_version" in payload
    assert "ensemble" in payload


def test_artifact_not_found_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_artifact("0" * 64, tmp_path)


def test_artifact_sha_mismatch_raises(tmp_path: Path) -> None:
    """Tamper detection: if the payload's content_sha256 doesn't match
    the filename, refuse to load."""
    # Synthesise a payload whose self-reported sha disagrees with where
    # we write it. We don't need a real booster — only the loader needs
    # to read enough to check the sha field.
    real_content = {"k": 1}
    real_sha = hashlib.sha256(
        json.dumps(real_content, sort_keys=True).encode()
    ).hexdigest()
    fake_sha = "1" + real_sha[1:]  # one-char tamper
    payload = {
        "content_sha256": real_sha,
        "spec": {"name": "synth"},
        "feature_names": [],
        "booster": None,
        "ensemble": None,
        "objective": "binary",
    }
    # Write at the FAKE sha's path so filename != self-reported sha.
    target = artifact_path(fake_sha, tmp_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(pickle.dumps(payload, protocol=5))
    with pytest.raises(ValueError, match="content_sha mismatch"):
        read_artifact(fake_sha, tmp_path)


def test_artifact_path_uses_two_char_shard(tmp_path: Path) -> None:
    sha = "abcdef" + "0" * 58
    p = artifact_path(sha, tmp_path)
    assert p.parent.name == "ab"
    assert p.name == f"{sha}.pkl"
