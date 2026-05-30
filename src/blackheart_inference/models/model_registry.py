"""Model registry for tracking versions, metadata, and deployment status.

Manages:
- Model registration with metadata (signal_id, artifact_path, feature_version, metrics)
- Deployment state (FULL or CANARY with percentage and fallback)
- Model retirement / status tracking
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any, Optional


class DeploymentType(Enum):
    """Deployment type enum."""

    FULL = "FULL"
    CANARY = "CANARY"


class ModelRegistry:
    """Registry for managing model versions and deployments."""

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._models: dict[str, dict[str, Any]] = {}
        self._deployments: dict[str, dict[str, Any]] = {}

    def register_model(
        self,
        model_id: str,
        signal_id: str,
        artifact_path: str,
        feature_version: str,
        metrics: Optional[dict[str, Any]] = None,
    ) -> None:
        """Register a model with metadata.

        Args:
            model_id: Unique model identifier
            signal_id: Signal this model predicts (e.g. regime_btc_v3)
            artifact_path: Path to pickled model artifact
            feature_version: Feature engineering version used
            metrics: Optional dict of metrics (auc, logloss, etc.)
        """
        if metrics is None:
            metrics = {}

        self._models[model_id] = {
            "model_id": model_id,
            "signal_id": signal_id,
            "artifact_path": artifact_path,
            "feature_version": feature_version,
            "metrics": metrics,
            "status": "active",
            "registered_at": datetime.now(UTC).isoformat(),
        }

    def get_model(self, model_id: str) -> Optional[dict[str, Any]]:
        """Get model metadata by ID.

        Args:
            model_id: Model identifier

        Returns:
            Model metadata dict or None if not found
        """
        return self._models.get(model_id)

    def deploy(
        self,
        model_id: str,
        deployment_type: DeploymentType,
        canary_percentage: int = 0,
        fallback_model_id: Optional[str] = None,
    ) -> None:
        """Deploy a model with specified type.

        Args:
            model_id: Model to deploy
            deployment_type: FULL or CANARY
            canary_percentage: For CANARY: percent traffic to canary (1-100)
            fallback_model_id: For CANARY: model to use for fallback traffic

        Raises:
            ValueError: If deployment config is invalid
        """
        if deployment_type == DeploymentType.CANARY:
            if fallback_model_id is None:
                raise ValueError(
                    "CANARY deployment requires fallback_model_id"
                )
            if canary_percentage <= 0 or canary_percentage > 99:
                raise ValueError(
                    "CANARY deployment requires canary_percentage in range (0, 99]"
                )

        self._deployments[model_id] = {
            "model_id": model_id,
            "deployment_type": deployment_type,
            "canary_percentage": canary_percentage,
            "fallback_model_id": fallback_model_id,
            "deployed_at": datetime.now(UTC).isoformat(),
        }

    def get_deployment(self, model_id: str) -> Optional[dict[str, Any]]:
        """Get deployment status by model ID.

        Args:
            model_id: Model identifier

        Returns:
            Deployment status dict or None if not deployed
        """
        return self._deployments.get(model_id)

    def retire_model(self, model_id: str) -> None:
        """Mark a model as retired.

        Args:
            model_id: Model to retire

        Raises:
            ValueError: If model not found
        """
        if model_id not in self._models:
            raise ValueError(f"Model {model_id} not found")

        self._models[model_id]["status"] = "retired"
