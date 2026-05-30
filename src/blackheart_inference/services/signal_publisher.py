"""Publish ML signals and inference errors to Kafka topics.

Publishes two message types:
- SignalPredictionEvent to signals.live (normal inference results)
- InferenceErrorEvent to errors.inference (failures and circuit breaker state)

Uses async Kafka producer for high throughput. Writes to signal_history table
are deferred to a future integration (TODO).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Optional

from aiokafka import AIOKafkaProducer

logger = logging.getLogger(__name__)


class SignalPublisher:
    """Publish inference results and errors to Kafka.

    Attributes:
        bootstrap_servers: Kafka bootstrap server addresses
        producer: AIOKafkaProducer instance (started on __aenter__)
    """

    def __init__(self, bootstrap_servers: list[str]) -> None:
        """Initialize signal publisher.

        Args:
            bootstrap_servers: List of Kafka bootstrap servers (e.g., ['localhost:9092'])
        """
        self.bootstrap_servers = bootstrap_servers
        self.producer: AIOKafkaProducer | None = None

    async def __aenter__(self) -> SignalPublisher:
        """Start the async Kafka producer."""
        self.producer = AIOKafkaProducer(bootstrap_servers=self.bootstrap_servers)
        await self.producer.start()
        logger.info(
            f"SignalPublisher started with bootstrap_servers={self.bootstrap_servers}"
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop the async Kafka producer."""
        if self.producer is not None:
            await self.producer.stop()
            logger.info("SignalPublisher stopped")

    async def publish_signal(
        self,
        signal_id: str,
        symbol: str,
        interval: str,
        ts: datetime,
        prediction: int,
        confidence: float,
        model_version: str,
        latency_ms: int,
    ) -> None:
        """Publish a successful inference result to signals.live topic.

        Creates a SignalPredictionEvent and publishes it to Kafka. Also logs
        to signal_history table (TODO).

        Args:
            signal_id: ML signal identifier (e.g., regime_btc_v3)
            symbol: Trading pair symbol (e.g., BTCUSDT)
            interval: Candle interval (1h, 4h, 1d, etc.)
            ts: Prediction timestamp (UTC)
            prediction: Model output (typically 0 or 1 for binary)
            confidence: Prediction confidence (0.0 to 1.0)
            model_version: Model version identifier (UUID or semantic version)
            latency_ms: Model inference latency in milliseconds

        Raises:
            RuntimeError: If producer not started or publishing fails
        """
        if self.producer is None:
            raise RuntimeError("SignalPublisher not started; use async with context")

        # Create the event payload (matches SignalPredictionEvent from Task 2)
        event = {
            "signal_id": signal_id,
            "symbol": symbol,
            "interval": interval,
            "ts": ts.isoformat() + "Z" if ts.tzinfo else ts.isoformat(),
            "prediction": prediction,
            "confidence": confidence,
            "model_version": model_version,
            "latency_ms": latency_ms,
        }

        # Publish to signals.live topic
        message = json.dumps(event).encode("utf-8")
        await self.producer.send_and_wait("signals.live", message)

        logger.debug(
            f"Published signal {signal_id}/{symbol}/{interval}: "
            f"prediction={prediction}, confidence={confidence:.3f}, latency_ms={latency_ms}"
        )

        # TODO: Write to signal_history table in PostgreSQL
        # This will be integrated with signal_history repo once available

    async def publish_error(
        self,
        signal_id: str,
        symbol: str,
        ts: datetime,
        error_type: str,
        error_message: str,
        circuit_breaker_state: str,
    ) -> None:
        """Publish an inference failure to errors.inference topic.

        Creates an InferenceErrorEvent and publishes it to Kafka for monitoring
        and alerting.

        Args:
            signal_id: ML signal identifier (e.g., regime_btc_v3)
            symbol: Trading pair symbol (e.g., BTCUSDT)
            ts: Error timestamp (UTC)
            error_type: Error category (inference_timeout, oom, model_error, etc.)
            error_message: Human-readable error description
            circuit_breaker_state: Circuit breaker state after error (closed, open, half_open)

        Raises:
            RuntimeError: If producer not started or publishing fails
        """
        if self.producer is None:
            raise RuntimeError("SignalPublisher not started; use async with context")

        # Create the event payload (matches InferenceErrorEvent from Task 2)
        event = {
            "signal_id": signal_id,
            "symbol": symbol,
            "ts": ts.isoformat() + "Z" if ts.tzinfo else ts.isoformat(),
            "error_type": error_type,
            "error_message": error_message,
            "circuit_breaker_state": circuit_breaker_state,
        }

        # Publish to errors.inference topic
        message = json.dumps(event).encode("utf-8")
        await self.producer.send_and_wait("errors.inference", message)

        logger.warning(
            f"Published error for {signal_id}/{symbol}: "
            f"error_type={error_type}, state={circuit_breaker_state}"
        )
