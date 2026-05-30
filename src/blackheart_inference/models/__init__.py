"""Model versioning and deployment management.

Exports:
- DeploymentType: Enum for FULL and CANARY deployments
- ModelRegistry: Registry for tracking model versions and deployments
- CanaryDeployer: Helper for canary A/B testing and promotion
"""

from blackheart_inference.models.model_registry import (
    DeploymentType,
    ModelRegistry,
)
from blackheart_inference.models.canary_deployment import CanaryDeployer

__all__ = [
    "DeploymentType",
    "ModelRegistry",
    "CanaryDeployer",
]
