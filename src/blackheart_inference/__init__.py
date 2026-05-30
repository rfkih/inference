"""Blackheart inference sidecar.

Closes the Phase 4 stage A3 architecture decision (Python sidecar arm):
artifact loaded from the same content-addressed store ``blackheart-train``
writes to, features fetched from ``feature_values``, predictions written
to ``signal_history`` so the trading JVM's ``MLSignalService`` can read
them without ever loading a LightGBM booster in-process.

Loopback-only HTTP boundary. The orchestrator (``:8082``) proxies the
researcher-facing surface; the trading JVM (``:8080``) and research JVM
(``:8081``) interact only through the shared database.
"""

__version__ = "0.1.0"
