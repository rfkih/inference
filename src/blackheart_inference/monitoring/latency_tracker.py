"""Latency tracker for monitoring inference performance against SLO targets.

Tracks p99 latency per signal and enforces <100ms SLA by default.
Maintains rolling window (max 10k per signal) to prevent memory bloat.
"""

from __future__ import annotations

from collections import defaultdict
from statistics import mean, quantiles
from typing import Any, Optional


class LatencyTracker:
    """Track p99 inference latency and enforce SLO targets."""

    def __init__(self, p99_slo_ms: float = 100.0, max_per_signal: int = 10000) -> None:
        """Initialize tracker with SLO threshold.

        Args:
            p99_slo_ms: P99 latency SLO in milliseconds (default 100ms)
            max_per_signal: Maximum measurements to retain per signal (default 10k)
        """
        self.p99_slo_ms = p99_slo_ms
        self._max_per_signal = max_per_signal
        self._measurements: dict[str, list[float]] = defaultdict(list)

    def record_latency(self, signal_id: str, latency_ms: float) -> None:
        """Record a latency measurement.

        Args:
            signal_id: Unique signal identifier (e.g. "regime_btc_v3")
            latency_ms: Measured latency in milliseconds

        Raises:
            ValueError: If latency is negative
        """
        if latency_ms < 0:
            raise ValueError(f"Latency cannot be negative: {latency_ms}")

        measurements = self._measurements[signal_id]
        measurements.append(latency_ms)

        # Enforce rolling window: keep only latest N measurements
        if len(measurements) > self._max_per_signal:
            # Drop oldest measurements
            excess = len(measurements) - self._max_per_signal
            del measurements[:excess]

    def get_stats(self, signal_id: str) -> Optional[dict[str, Any]]:
        """Get latency statistics for a signal.

        Args:
            signal_id: Signal identifier

        Returns:
            Dict with {count, min, max, mean, p50, p99} or None if no data
        """
        if signal_id not in self._measurements:
            return None

        measurements = self._measurements[signal_id]
        if not measurements:
            return None

        sorted_measurements = sorted(measurements)
        count = len(sorted_measurements)

        # Calculate percentiles using quantiles
        # For n samples, quantiles(data, n=100) returns 100 equally-spaced quantiles
        # quantiles handles edge cases (single value, etc.)
        if count == 1:
            p50 = sorted_measurements[0]
            p99 = sorted_measurements[0]
        elif count == 2:
            # For 2 samples, p50 is between them, p99 is the higher
            p50 = (sorted_measurements[0] + sorted_measurements[1]) / 2
            p99 = sorted_measurements[1]
        else:
            # Use quantiles to compute p50 and p99
            # quantiles(data, n=100) gives 99 values dividing the data into 100 equal groups
            q = quantiles(sorted_measurements, n=100)
            p50 = q[49]  # 50th percentile (0-indexed: position 49 of 99)
            p99 = q[98]  # 99th percentile (0-indexed: position 98 of 99)

        return {
            "count": count,
            "min": min(sorted_measurements),
            "max": max(sorted_measurements),
            "mean": mean(sorted_measurements),
            "p50": p50,
            "p99": p99,
        }

    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """Get stats for all tracked signals.

        Returns:
            Dict mapping signal_id -> stats dict
        """
        result = {}
        for signal_id in self._measurements.keys():
            stats = self.get_stats(signal_id)
            if stats is not None:
                result[signal_id] = stats
        return result

    def check_slo_violation(self, signal_id: str) -> bool:
        """Check if signal's p99 latency exceeds SLO.

        Args:
            signal_id: Signal identifier

        Returns:
            True if p99 > SLO threshold, False otherwise (also False if no data)
        """
        stats = self.get_stats(signal_id)
        if stats is None:
            return False
        return stats["p99"] > self.p99_slo_ms

    def reset(self, signal_id: Optional[str] = None) -> None:
        """Clear measurements.

        Args:
            signal_id: If provided, reset only this signal. If None, reset all.
        """
        if signal_id is None:
            self._measurements.clear()
        elif signal_id in self._measurements:
            del self._measurements[signal_id]
