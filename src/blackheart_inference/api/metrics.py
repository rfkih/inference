"""``/metrics/latency`` endpoint — P99 inference latency and SLO tracking."""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from ..monitoring.latency_tracker import LatencyTracker

router = APIRouter(prefix="/metrics", tags=["metrics"])


class LatencyMetrics(BaseModel):
    """Latency statistics for a single signal."""

    signal_id: str = Field(..., description="Signal identifier")
    count: int = Field(..., description="Number of measurements")
    min: float = Field(..., description="Minimum latency (ms)")
    max: float = Field(..., description="Maximum latency (ms)")
    mean: float = Field(..., description="Mean latency (ms)")
    p50: float = Field(..., description="50th percentile (median) latency (ms)")
    p99: float = Field(..., description="99th percentile latency (ms)")
    slo_compliant: bool = Field(
        ..., description="True if p99 <= SLO threshold, False otherwise"
    )


class MlMetricsResponse(BaseModel):
    """Response for metrics endpoints."""

    metrics: list[LatencyMetrics] = Field(
        default_factory=list, description="List of latency metrics per signal"
    )
    p99_slo_ms: float = Field(..., description="P99 latency SLO threshold (ms)")


async def get_latency_tracker(request: Request) -> LatencyTracker:
    """Dependency to get the latency tracker from app state."""
    if not hasattr(request.app.state, "latency_tracker"):
        # Initialize on first use
        request.app.state.latency_tracker = LatencyTracker()
    return request.app.state.latency_tracker


@router.get("/latency", response_model=MlMetricsResponse)
async def get_latency_metrics(
    tracker: LatencyTracker = Depends(get_latency_tracker),
) -> MlMetricsResponse:
    """Get current latency metrics for all signals.

    Returns:
        MlMetricsResponse with list of LatencyMetrics and SLO threshold
    """
    all_stats = tracker.get_all_stats()

    metrics_list = []
    for signal_id, stats in all_stats.items():
        slo_compliant = not tracker.check_slo_violation(signal_id)
        metrics_list.append(
            LatencyMetrics(
                signal_id=signal_id,
                count=stats["count"],
                min=stats["min"],
                max=stats["max"],
                mean=stats["mean"],
                p50=stats["p50"],
                p99=stats["p99"],
                slo_compliant=slo_compliant,
            )
        )

    return MlMetricsResponse(metrics=metrics_list, p99_slo_ms=tracker.p99_slo_ms)


@router.post("/reset")
async def reset_metrics(
    signal_id: Optional[str] = None,
    tracker: LatencyTracker = Depends(get_latency_tracker),
) -> dict[str, Any]:
    """Reset latency metrics.

    Args:
        signal_id: If provided, reset only this signal. If None, reset all.

    Returns:
        Confirmation message
    """
    if signal_id:
        tracker.reset(signal_id)
        return {"status": "ok", "message": f"Reset metrics for signal {signal_id}"}
    else:
        tracker.reset()
        return {"status": "ok", "message": "Reset all metrics"}
