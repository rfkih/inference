"""Per-signal circuit breaker for inference fault tolerance.

Implements the standard three-state circuit breaker pattern:
- CLOSED: Normal operation, accepting all requests
- OPEN: Failing, rejecting all requests until recovery timeout elapses
- HALF_OPEN: Testing recovery, allowing one request to probe health

Transitions:
  CLOSED -> OPEN: When failure_count reaches failure_threshold
  OPEN -> HALF_OPEN: Automatic after recovery_timeout_secs elapses
  HALF_OPEN -> CLOSED: On success (reset failure count)
  HALF_OPEN -> OPEN: On failure (re-open immediately)

Logging emits on all state transitions for monitoring and alerting.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from enum import Enum
from time import time

logger = logging.getLogger(__name__)


class CircuitBreakerState(str, Enum):
    """States of the circuit breaker."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Per-signal circuit breaker for inference fault tolerance.

    Tracks consecutive failures and automatically transitions between
    states to prevent cascading failures from a bad model or hung process.

    Attributes:
        signal_id: Unique identifier for the signal (e.g., regime_btc_v3)
        failure_count: Number of consecutive failures in current CLOSED period
        last_failure_time: Timestamp of the most recent failure
        _state: Current circuit breaker state
        failure_threshold: Number of failures before opening
        recovery_timeout_secs: Seconds to wait before transitioning OPEN → HALF_OPEN
    """

    def __init__(
        self,
        signal_id: str,
        failure_threshold: int = 5,
        recovery_timeout_secs: int = 300,
    ) -> None:
        """Initialize circuit breaker for a signal.

        Args:
            signal_id: Unique signal identifier
            failure_threshold: Number of failures before opening (default 5)
            recovery_timeout_secs: Recovery timeout in seconds (default 300s)
        """
        self.signal_id = signal_id
        self.failure_count = 0
        self.last_failure_time: float | None = None
        self._state = CircuitBreakerState.CLOSED
        self.failure_threshold = failure_threshold
        self.recovery_timeout_secs = recovery_timeout_secs

    @property
    def state(self) -> CircuitBreakerState:
        """Get current state with automatic timeout-based transitions.

        Automatically transitions OPEN → HALF_OPEN when recovery_timeout_secs
        has elapsed since the last failure.
        """
        # If OPEN and timeout elapsed, transition to HALF_OPEN
        if self._state == CircuitBreakerState.OPEN and self.last_failure_time is not None:
            elapsed = time() - self.last_failure_time
            if elapsed >= self.recovery_timeout_secs:
                self._state = CircuitBreakerState.HALF_OPEN
                logger.warning(
                    f"Circuit breaker for {self.signal_id} transitioned OPEN -> HALF_OPEN "
                    f"(recovery timeout {self.recovery_timeout_secs}s elapsed)"
                )
        return self._state

    def record_failure(self) -> None:
        """Record a failure and transition to OPEN if threshold exceeded.

        If in HALF_OPEN, immediately reopen on any failure.
        If in CLOSED, increment count and transition to OPEN if threshold met.
        """
        self.failure_count += 1
        self.last_failure_time = time()

        if self._state == CircuitBreakerState.HALF_OPEN:
            self._state = CircuitBreakerState.OPEN
            logger.warning(
                f"Circuit breaker for {self.signal_id} transitioned HALF_OPEN -> OPEN "
                f"(failure in recovery testing)"
            )
        elif self._state == CircuitBreakerState.CLOSED:
            if self.failure_count >= self.failure_threshold:
                self._state = CircuitBreakerState.OPEN
                logger.warning(
                    f"Circuit breaker for {self.signal_id} transitioned CLOSED -> OPEN "
                    f"(failure_count {self.failure_count} >= threshold {self.failure_threshold})"
                )

    def record_success(self) -> None:
        """Record a success and potentially transition to CLOSED.

        If in HALF_OPEN, transition to CLOSED (recovery successful).
        If in CLOSED, reset failure count.
        If in OPEN, do nothing (wait for recovery timeout).
        """
        if self._state == CircuitBreakerState.HALF_OPEN:
            self._state = CircuitBreakerState.CLOSED
            self.failure_count = 0
            logger.info(
                f"Circuit breaker for {self.signal_id} transitioned HALF_OPEN -> CLOSED "
                f"(recovery successful)"
            )
        elif self._state == CircuitBreakerState.CLOSED:
            self.failure_count = 0

    def reset(self) -> None:
        """Manually reset the circuit breaker to CLOSED with zero failures.

        Useful for explicit administrative recovery or testing.
        """
        self._state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        logger.info(f"Circuit breaker for {self.signal_id} manually reset to CLOSED")

    def is_available(self) -> bool:
        """Check if the circuit breaker allows requests.

        Returns True if in CLOSED or HALF_OPEN state (accepting requests),
        False if in OPEN state (rejecting all requests).
        """
        current_state = self.state  # Call property to trigger auto-transition
        return current_state in (
            CircuitBreakerState.CLOSED,
            CircuitBreakerState.HALF_OPEN,
        )
