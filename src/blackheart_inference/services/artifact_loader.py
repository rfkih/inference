"""Load a blackheart-train artifact by content_sha256.

Mirrors ``blackheart_train.artifacts.read_artifact`` byte-for-byte so
this service can read every artifact ever written without depending on
the training package as an import. We re-implement rather than depend
because:

  * blackheart-train ships ``lightgbm`` + ``scikit-learn`` + the heavy
    feature transformer stack. Inference only needs LightGBM and the
    artifact loader.
  * Decoupling the two services means a training-only refactor cannot
    break inference at the symbol level. The artifact format is the
    contract, not the code.

Artifact path layout: ``<artifact_dir>/<sha256[:2]>/<sha256>.pkl``.

Payload shape (verified by ``content_sha256`` round-trip):

  * ``content_sha256``    str — must match filename's sha
  * ``payload_version``   int — 1 or 2 (v2 carries optional ``ensemble``)
  * ``spec``              dict — ModelSpec frozen-dataclass dump
  * ``feature_names``     list[str] — input column order
  * ``booster``           lgb.Booster or None (None for ensembles)
  * ``ensemble``          dict or None (v2 ensembles)
  * ``objective``         'binary' | 'regression' | 'multiclass'
  * ``label_feature``     str
  * ``label_version``     int

Forward-compat with v1 artifacts (no ``payload_version`` key): we
backfill ``payload_version=1, ensemble=None`` so consumers can branch
on the version uniformly.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def artifact_path(content_sha: str, artifact_dir: Path) -> Path:
    """The on-disk location for a given content_sha."""
    return artifact_dir / content_sha[:2] / f"{content_sha}.pkl"


def read_artifact(content_sha: str, artifact_dir: Path) -> dict[str, Any]:
    """Load an artifact, verify the round-trip sha, return the payload.

    Raises ``FileNotFoundError`` if the path doesn't exist. Raises
    ``ValueError`` if the payload's self-reported ``content_sha256`` does
    not match the filename — that's a tampering signal we refuse rather
    than silently use.
    """
    path = artifact_path(content_sha, artifact_dir)
    if not path.exists():
        raise FileNotFoundError(f"artifact not found: {path}")
    payload = pickle.loads(path.read_bytes())
    stored = payload.get("content_sha256")
    if stored != content_sha:
        raise ValueError(
            f"artifact content_sha mismatch at {path}: "
            f"filename says {content_sha}, payload says {stored!r}"
        )
    payload.setdefault("payload_version", 1)
    payload.setdefault("ensemble", None)
    return payload
