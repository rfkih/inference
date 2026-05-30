# blackheart-inference

ML inference sidecar for Blackheart. Closes the Phase 4 stage A3 architecture decision (Python sidecar arm): loads model_registry artifacts trained by `blackheart-train`, fetches features from `feature_values`, runs `Booster.predict()`, writes `signal_history` rows the trading JVM consumes via `MLSignalService` and `MLRegimeGateGuard`.

## Topology

```
  blackheart-train  ─► artifacts/<sha[:2]>/<sha>.pkl ─┐
                                                     │
quant-researcher ─► orchestrator ─► /inference/* ────┼─► blackheart-inference :8000
                                                     │       │
                                                     │       ▼ feature_values + model_registry
                                                     │      PostgreSQL
                                                     │       ▲ signal_history
                                                     │       │
                                                     └──► trading JVM :8080 (MLSignalService reads)
```

Loopback-only. Direct callers: the orchestrator and (eventually) the live-streaming worker cron. The researcher never talks to `:8000` directly — only through the orchestrator's `/inference/run` and `/inference/backfill` proxies.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/healthz` | Process is up. Public. |
| `GET`  | `/readyz` | DB + artifact_dir reachable. Public. |
| `POST` | `/inference/run` | Single-bar inference. Resolves signal → model → artifact → feature row → predict → upsert. |
| `POST` | `/inference/backfill` | Window inference. Used by the researcher to populate `signal_history` over a historical range for paired-backtests. |

`/inference/run` body:

```json
{
  "signal_id": "<uuid>",
  "symbol": "BTCUSDT",
  "ts": "2026-05-01T00:00:00Z",
  "interval_name": "1h",
  "source": "stream"
}
```

`/inference/backfill` body:

```json
{
  "signal_id": "<uuid>",
  "symbol": "BTCUSDT",
  "interval_name": "1h",
  "start": "2024-01-01T00:00:00Z",
  "end":   "2026-05-01T00:00:00Z",
  "source": "historical_replay"
}
```

Returns the row count written and rows skipped due to missing features. `signal_history.source` is stamped per the request.

## Quick start (dev)

```powershell
cd C:\Project\blackheart-inference
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
Copy-Item .env.example .env   # then edit INFERENCE_AUTH_TOKEN, INFERENCE_DB_DSN
python -m blackheart_inference   # serves on 127.0.0.1:8000
```

In another shell:

```bash
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/readyz   # checks DB + artifact_dir
```

## Hard contract — DO NOT change without user approval

| Boundary | Why frozen |
|---|---|
| Artifact format (`content_sha256` round-trip, pickle protocol 5, payload keys) | Cross-service contract with blackheart-train. Changing format breaks every registered artifact. |
| `signal_history.source` enum (`stream` \| `catchup_scan` \| `historical_replay`) | V66 CHECK constraint. Adding a value requires Flyway. |
| Loopback bind (`INFERENCE_HOST=127.0.0.1`) | Security boundary. Token is defense-in-depth. |
| `INFERENCE_AUTH_TOKEN` dev sentinel refused under `INFERENCE_PROFILE=prod` | Matches the orchestrator's `assert_prod_safe` pattern — fails fast at startup. |

## Editable surface

- `src/blackheart_inference/` — handlers / services / repo / clients
- `tests/` — unit + integration
- `pyproject.toml` — deps as needed (e.g. for additional model formats)
- `README.md` — keep honest after API changes

## Conventions

### asyncpg + JSONB

The codec in `infra/db.py` registers JSONB as dict — pass dicts directly to `$N` params (do NOT `json.dumps` first). Matches the orchestrator's convention.

### Error envelope

Shape-identical to the orchestrator's `ErrorEnvelope` so the orchestrator's `InferenceClient` can forward errors verbatim. Callers (the researcher, via the orchestrator) branch on `error_code` — never on prose.

### Refusing to serve

The service refuses inference on:

* Signal status `'retired'` → 409 `signal_retired`
* Model status `'retired'` or `'rejected_by_operator'` → 409 `model_lifecycle_refused`
* Model with NULL `artifact_sha256` → 409 `model_artifact_missing`
* Missing feature in `feature_registry` (`status != 'registered'`) → 409 `feature_not_registered`
* Missing feature_values rows at the requested ts → 409 `feature_value_missing` (run a feature backfill first)
* Backfill window producing more rows than `INFERENCE_MAX_BACKFILL_ROWS` → 413 `backfill_window_too_large`
* Artifact sha mismatch on disk → 502 `artifact_corrupt` (tamper detection)

## What's NOT here yet

| Capability | Status | Where it goes |
|---|---|---|
| Live-streaming worker (cron-fires inference on every new bar) | Not built | `services/streaming.py` + cron entry |
| Paired-backtest harness (`POST /paired-backtest` — WITH vs WITHOUT ML gate) | Not built | Orchestrator-side; uses this service's `/inference/backfill` + JVM submit twice |
| Ensemble artifact support (`payload['ensemble'] != None`) | Refused with `NotImplementedError` | `services/predictor.py` |
| Multiclass post-processing beyond argmax | Argmax only | `services/predictor.py:predict_single` |
| Remote artifact_uri (S3 / GCS) | Local-FS only | `services/artifact_loader.py` |

## Tests

```bash
.\.venv\Scripts\python.exe -m pytest -q
```

16 tests covering: auth middleware, artifact loader round-trip + tamper detection, predictor contract (feature order, missing-feature error, batch skipping), endpoint validation envelope shape. DB-touching paths are not exercised here — those need pytest-postgresql + real migrations and belong in an integration suite.

## Links

- `../blackheart-train/README.md` — training side
- `../blackheart-research-orchestrator/CLAUDE.md` — orchestrator (proxy lives here)
- `../blackheart-trading-engine/src/main/java/id/co/blackheart/service/mlsignal/MLSignalService.java` — JVM consumer
- `../blackheart-trading-engine/src/main/java/id/co/blackheart/service/risk/MLRegimeGateGuard.java` — JVM risk gate
- V66 migration: `../blackheart-trading-engine/src/main/resources/db/flyway/V66__add_ml_sentiment_schema.sql`
