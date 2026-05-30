"""Consume feature events from Kafka and run inference with circuit breaker.

Main flow:
1. Read ComputedFeatureEvent from features.computed topic
2. Run inference with circuit breaker protection
3. On success: publish SignalPredictionEvent to signals.live
4. On error: publish InferenceErrorEvent to errors.inference

Circuit breaker prevents cascading failures from hung models by:
- Opening after N consecutive failures
- Preventing inference attempts while OPEN
- Auto-transitioning to HALF_OPEN after timeout for recovery testing
- Closing on successful recovery
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from time import time
from typing import Any, Callable

from aiokafka import AIOKafkaConsumer

from blackheart_inference.services.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerState,
)
from blackheart_inference.services.signal_publisher import SignalPublisher

logger = logging.getLogger(__name__)


class InferenceConsumer:
    """Consume feature events and run inference with circuit breaker protection.

    Attributes:
        bootstrap_servers: Kafka bootstrap server addresses
        predict_fn: Callable that takes feature_vector dict → (prediction, confidence)
        signal_id: Unique signal identifier (e.g., regime_btc_v3)
        model_version: Model version identifier (UUID or semantic version)
        circuit_breaker: Per-signal circuit breaker instance
        failure_threshold: Circuit breaker failure threshold
        recovery_timeout_secs: Circuit breaker recovery timeout
        consumer: AIOKafkaConsumer instance (started on start())
        publisher: SignalPublisher instance (started on start())
    """

    def __init__(
        self,
        bootstrap_servers: list[str],
        predict_fn: Callable[[dict[str, float]], tuple[int, float]],
        signal_id: str,
        model_version: str,
        failure_threshold: int = 5,
        recovery_timeout_secs: int = 300,
    ) -> None:
        """Initialize inference consumer.

        Args:
            bootstrap_servers: List of Kafka bootstrap servers
            predict_fn: Function that runs inference (feature_vector → (pred, conf))
            signal_id: Unique signal identifier
            model_version: Model version identifier
            failure_threshold: Circuit breaker failure threshold (default 5)
            recovery_timeout_secs: Circuit breaker recovery timeout (default 300s)
        """
        self.bootstrap_servers = bootstrap_servers
        self.predict_fn = predict_fn
        self.signal_id = signal_id
        self.model_version = model_version
        self.failure_threshold = failure_threshold
        self.recovery_timeout_secs = recovery_timeout_secs

        # Initialize circuit breaker for this signal
        self.circuit_breaker = CircuitBreaker(
            signal_id=signal_id,
            failure_threshold=failure_threshold,
            recovery_timeout_secs=recovery_timeout_secs,
        )

        self.consumer: AIOKafkaConsumer | None = None
        self.publisher: SignalPublisher | None = None

    async def start(self) -> None:
        """Start the Kafka consumer and signal publisher.

        Creates AIOKafkaConsumer for features.computed topic and starts
        the SignalPublisher for publishing results.
        """
        # Create and start the consumer
        self.consumer = AIOKafkaConsumer(
            "features.computed",
            bootstrap_servers=self.bootstrap_servers,
            group_id=f"inference-{self.signal_id}",
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            auto_offset_reset="latest",
        )
        await self.consumer.start()
        logger.info(
            f"InferenceConsumer started for {self.signal_id} "
            f"with bootstrap_servers={self.bootstrap_servers}"
        )

        # Create and start the publisher
        self.publisher = SignalPublisher(bootstrap_servers=self.bootstrap_servers)
        await self.publisher.__aenter__()

    async def stop(self) -> None:
        """Stop the Kafka consumer and signal publisher."""
        if self.consumer is not None:
            await self.consumer.stop()
            logger.info(f"InferenceConsumer stopped for {self.signal_id}")

        if self.publisher is not None:
            await self.publisher.__aexit__(None, None, None)

    async def _run_inference(self, feature_vector: dict[str, float]) -> tuple[int, float]:
        """Run inference on a feature vector with circuit breaker protection.

        Calls predict_fn and records success/failure on the circuit breaker.
        Raises RuntimeError if circuit is OPEN (rejecting requests).

        Args:
            feature_vector: Dictionary of feature name → float value

        Returns:
            Tuple of (prediction, confidence)

        Raises:
            RuntimeError: If circuit breaker is OPEN or predict_fn fails
        """
        # Check circuit breaker — reject if OPEN
        if not self.circuit_breaker.is_available():
            raise RuntimeError(
                f"Circuit breaker for {self.signal_id} is {self.circuit_breaker.state} "
                f"— inference requests rejected"
            )

        # Run inference
        try:
            start_time = time()
            prediction, confidence = self.predict_fn(feature_vector)
            elapsed_ms = int((time() - start_time) * 1000)

            # Record success
            self.circuit_breaker.record_success()

            return prediction, confidence

        except Exception as e:
            # Record failure and re-raise
            self.circuit_breaker.record_failure()
            raise RuntimeError(f"Inference failed: {str(e)}") from e

    async def run(self) -> None:
        """Main consumer loop: read features, run inference, publish signals.

        Reads ComputedFeatureEvent from features.computed topic, runs inference
        with circuit breaker protection, and publishes results to signals.live
        or errors.inference as appropriate.

        Intended to run indefinitely; should be run in a background task.
        """
        if self.consumer is None or self.publisher is None:
            raise RuntimeError("Consumer not started; call start() first")

        logger.info(f"InferenceConsumer.run() starting for {self.signal_id}")

        try:
            async for message in self.consumer:
                try:
                    # Parse the feature event
                    feature_event = message.value

                    # Only process events for this signal
                    if feature_event.get("signal_id") != self.signal_id:
                        continue

                    symbol = feature_event.get("symbol", "UNKNOWN")
                    interval = feature_event.get("interval", "UNKNOWN")
                    ts_str = feature_event.get("ts", datetime.now(timezone.utc).isoformat())
                    feature_vector = feature_event.get("feature_vector", {})

                    # Parse timestamp
                    try:
                        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    except (ValueError, AttributeError):
                        ts = datetime.now(timezone.utc)

                    # Run inference with circuit breaker
                    try:
                        start_time = time()
                        prediction, confidence = await self._run_inference(feature_vector)
                        latency_ms = int((time() - start_time) * 1000)

                        # Publish successful prediction
                        await self.publisher.publish_signal(
                            signal_id=self.signal_id,
                            symbol=symbol,
                            interval=interval,
                            ts=ts,
                            prediction=prediction,
                            confidence=confidence or 0.5,
                            model_version=self.model_version,
                            latency_ms=latency_ms,
                        )

                    except RuntimeError as e:
                        # Publish error event
                        await self.publisher.publish_error(
                            signal_id=self.signal_id,
                            symbol=symbol,
                            ts=datetime.now(timezone.utc),
                            error_type="inference_failure",
                            error_message=str(e),
                            circuit_breaker_state=self.circuit_breaker.state.value,
                        )
                        logger.error(f"Inference error for {self.signal_id}/{symbol}: {e}")

                except Exception as e:
                    logger.error(f"Error processing feature event: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"Consumer loop error for {self.signal_id}: {e}", exc_info=True)
            raise
