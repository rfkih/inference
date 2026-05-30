"""Feature store with schema versioning and lineage tracking.

In-memory store for computed features with schema versioning support.
Enables reproducibility by pinning to specific feature versions.

Production implementation would use InfluxDB or TimescaleDB.
"""

from __future__ import annotations

from typing import Any


class FeatureStore:
    """In-memory feature store with schema versioning.

    Stores computed features keyed by (signal_id, symbol) with schema validation
    against registered versions. Supports feature lineage tracking by pinning
    to specific schema versions for reproducibility.

    Attributes:
        _schemas: Dict of version -> schema metadata
        _features: Dict of (signal_id, symbol) -> List[feature_records]
    """

    def __init__(self) -> None:
        """Initialize feature store."""
        self._schemas: dict[str, dict[str, Any]] = {}
        self._features: dict[tuple[str, str], list[dict[str, Any]]] = {}

    def register_schema(
        self, version: str, feature_names: list[str]
    ) -> None:
        """Register a feature schema version.

        Args:
            version: Schema version identifier (e.g., "v1", "v2")
            feature_names: List of exact feature column names for this version

        Raises:
            ValueError: If version already registered with different feature names
        """
        if version in self._schemas:
            existing = self._schemas[version]
            if existing["feature_names"] != feature_names:
                raise ValueError(
                    f"Schema version {version} already registered "
                    f"with different feature names"
                )
            return

        self._schemas[version] = {
            "version": version,
            "feature_names": feature_names,
        }

    def get_schema(self, version: str) -> dict[str, Any] | None:
        """Retrieve registered schema by version.

        Args:
            version: Schema version identifier

        Returns:
            Schema metadata dict or None if not registered
        """
        return self._schemas.get(version)

    def store_features(
        self,
        signal_id: str,
        symbol: str,
        ts: int,
        features: dict[str, float],
        feature_version: str,
    ) -> None:
        """Store computed features with schema validation.

        Validates that feature dict matches registered schema version.
        Stores with timestamp for temporal tracking.

        Args:
            signal_id: Signal identifier (e.g., "regime_btc_v3")
            symbol: Trading symbol (e.g., "BTCUSDT")
            ts: Timestamp (seconds since epoch)
            features: Feature values dict keyed by feature name
            feature_version: Schema version to validate against

        Raises:
            ValueError: If version not registered or features don't match schema
        """
        # Validate schema exists
        schema = self.get_schema(feature_version)
        if schema is None:
            raise ValueError(
                f"Schema version {feature_version} not registered"
            )

        # Validate feature names match schema exactly
        expected_names = set(schema["feature_names"])
        actual_names = set(features.keys())

        if expected_names != actual_names:
            raise ValueError(
                f"Feature names do not match schema for version {feature_version}. "
                f"Expected {sorted(expected_names)}, got {sorted(actual_names)}"
            )

        # Store with metadata
        key = (signal_id, symbol)
        if key not in self._features:
            self._features[key] = []

        record = {
            **features,
            "__ts__": ts,
            "__version__": feature_version,
        }
        self._features[key].append(record)

    def get_features(
        self, signal_id: str, symbol: str
    ) -> dict[str, float] | None:
        """Retrieve latest features for signal/symbol pair.

        Returns the most recently stored feature record.

        Args:
            signal_id: Signal identifier
            symbol: Trading symbol

        Returns:
            Latest feature dict with __ts__ and __version__ metadata,
            or None if no features stored
        """
        key = (signal_id, symbol)
        records = self._features.get(key)

        if not records:
            return None

        # Return latest (last) record
        return records[-1]

    def get_feature_lineage(
        self, signal_id: str, symbol: str, feature_version: str
    ) -> dict[str, float] | None:
        """Retrieve features matching specific schema version.

        Enables reproducibility by pinning to a historical schema version,
        even if newer versions are available.

        Args:
            signal_id: Signal identifier
            symbol: Trading symbol
            feature_version: Schema version to retrieve

        Returns:
            Feature record matching schema version with metadata,
            or None if not found
        """
        key = (signal_id, symbol)
        records = self._features.get(key)

        if not records:
            return None

        # Find most recent record matching version
        for record in reversed(records):
            if record.get("__version__") == feature_version:
                return record

        return None
