"""Canary deployment helper for A/B testing and promotion.

Provides:
- choose_model: Route traffic between canary and fallback based on percentage
- promote_canary: Move canary to full deployment
- rollback_canary: Retire canary model on issues
"""

from __future__ import annotations

import hashlib
from typing import Optional

from blackheart_inference.models.model_registry import (
    DeploymentType,
    ModelRegistry,
)


class CanaryDeployer:
    """Helper for managing canary deployments and A/B testing."""

    def __init__(self, registry: ModelRegistry) -> None:
        """Initialize canary deployer.

        Args:
            registry: ModelRegistry instance to use for model metadata
        """
        self.registry = registry

    def choose_model(
        self,
        signal_id: str,
        primary_model_id: str,
        request_id: Optional[str] = None,
    ) -> str:
        """Choose which model to use for inference.

        Routes to primary model if FULL deployment, or uses deterministic
        hash-based routing for CANARY deployments (consistent per request_id).

        Args:
            signal_id: Signal being predicted
            primary_model_id: Primary/canary model ID
            request_id: Optional request ID for deterministic routing

        Returns:
            Model ID to use for inference
        """
        deployment = self.registry.get_deployment(primary_model_id)

        # No deployment → use primary
        if deployment is None:
            return primary_model_id

        # FULL deployment → use primary
        if deployment["deployment_type"] == DeploymentType.FULL:
            return primary_model_id

        # CANARY deployment → use percentage-based routing
        canary_percentage = deployment["canary_percentage"]
        fallback_model_id = deployment["fallback_model_id"]

        # Deterministic routing: hash request_id (or signal_id) to decide
        if request_id is None:
            request_id = signal_id

        hash_val = hashlib.sha256(request_id.encode()).hexdigest()
        hash_int = int(hash_val, 16)
        routing_percent = (hash_int % 100) + 1  # 1-100 inclusive

        if routing_percent <= canary_percentage:
            return primary_model_id
        else:
            return fallback_model_id

    def promote_canary(self, model_id: str) -> None:
        """Promote canary model to full deployment.

        Converts CANARY deployment to FULL deployment (100% traffic).

        Args:
            model_id: Canary model to promote

        Raises:
            ValueError: If model not deployed or not in CANARY state
        """
        deployment = self.registry.get_deployment(model_id)

        if deployment is None:
            raise ValueError(f"Model {model_id} has no deployment")

        if deployment["deployment_type"] != DeploymentType.CANARY:
            raise ValueError(f"Cannot promote non-CANARY model {model_id}")

        # Re-deploy as FULL
        self.registry.deploy(
            model_id=model_id,
            deployment_type=DeploymentType.FULL,
            canary_percentage=0,
            fallback_model_id=None,
        )

    def rollback_canary(self, model_id: str) -> None:
        """Rollback canary model due to issues.

        Retires the canary model, reverting to fallback.

        Args:
            model_id: Canary model to rollback

        Raises:
            ValueError: If model not deployed or not in CANARY state
        """
        deployment = self.registry.get_deployment(model_id)

        if deployment is None:
            raise ValueError(f"Model {model_id} has no deployment")

        if deployment["deployment_type"] != DeploymentType.CANARY:
            raise ValueError(f"Cannot rollback non-CANARY model {model_id}")

        # Retire the model
        self.registry.retire_model(model_id)
