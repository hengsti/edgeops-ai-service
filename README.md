# edgeops-ai-service

Anomaly detection for the EdgeOps smart-home stack. The service watches pipeline
metrics coming from the edge collector, classifies the current state as normal or
one of four fault types, and tells you what to check first.

Fault labels it knows about:

- `ingestion-backpressure` – the queue grows faster than messages get processed
- `missing-device-heartbeat` – a device went silent
- `malformed-sensor-payload` – payloads fail transformation and land in the DLQ
- `influxdb-write-failure` – data gets transformed but never persisted

There are two interchangeable prediction backends: a hand-written rule baseline and
a small scikit-learn classifier. More on why below.

## How it works

```
 collector (metrics API)
        |
        | polling                     POST /v1/observations
        v                                     |
 +------------------------------------------------+
 |               edgeops-ai-service               |
 |                                                |
 |   FeatureExtractor                             |
 |   (rates + ratios from two consecutive         |
 |    observations)                               |
 |          |                                     |
 |          v                                     |
 |   Predictor:  rules  -or-  ml                  |
 +------------------------------------------------+
        |
        v
 GET /v1/detections/latest
```

Observations reach the service either through the built-in collector polling loop
or by pushing them to `POST /v1/observations`. The feature extractor needs two
consecutive snapshots to compute rates, so the first response after startup is
always `"status": "warming-up"`.

## Quickstart

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```powershell
uv sync
uv run uvicorn edgeops_ai.main:create_app --factory --reload
```

Or with Docker:

```powershell
docker build -t edgeops-ai-service .
docker run --rm -p 8080:8080 -e PREDICTION_BACKEND=ml edgeops-ai-service
```

Configuration is plain environment variables (a `.env` file works too):

| Variable | Default | Purpose |
| --- | --- | --- |
| `PREDICTION_BACKEND` | `rules` | `rules` or `ml` |
| `MODEL_ARTIFACT_PATH` | `artifacts/model.joblib` | trained model, only used by the ml backend |
| `COLLECTOR_BASE_URL` | `http://localhost:8095` | collector metrics API |
| `COLLECTOR_API_KEY` | `local-dev-key` | sent as `X-EdgeOps-Key` |
| `COLLECTOR_POLLING_ENABLED` | `true` | disable if you only push observations |
| `COLLECTOR_POLL_INTERVAL_SECONDS` | `10` | polling interval |
| `COLLECTOR_TIMEOUT_SECONDS` | `3` | request timeout |
| `COLLECTOR_DEVICE_IDS` | *(empty)* | comma-separated devices for heartbeat checks |

## API

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | liveness plus active backend and model version |
| `POST` | `/v1/observations` | analyze one observation, returns the result |
| `GET` | `/v1/detections/latest` | most recent analysis (404 before the first one) |

A detection looks like this:

```json
{
  "status": "predicted",
  "captured_at": "2024-01-01T02:03:03+01:00",
  "prediction": "ingestion-backpressure",
  "confidence": 0.98,
  "model_version": "lightweight-ml-0.2.0",
  "recommendation": "Inspect ingestion queue pressure and processing throughput.",
  "features": { "queue_growth_rate": 14.2, "processing_ratio": 0.61, "...": "..." }
}
```

## Rules first, then ML

I deliberately started with a rule-based classifier (`src/edgeops_ai/rules.py`).
Five thresholds, readable in one screen, easy to debug against the collector's
simulation scenarios. That baseline reaches 99.6 % accuracy on stable phases
across 80 simulated runs (`data/generated/rule-baseline-report.json`) - and more
importantly, it produced labeled training data.

The ML backend is a scikit-learn model trained on exactly that data. Two
candidates get compared with leave-one-seed-out cross-validation (logistic
regression and a depth-6 decision tree); the tree won. Features are
workload-independent ratios rather than raw counters, so the model doesn't just
memorize the simulation's message rates.

On five completely unseen simulation seeds, evaluated end-to-end through the
deployed API, the model classified all 1120 stable-phase samples correctly and
flagged every fault at the first sample of the fault phase
(`data/holdout/deployed-ml-report.json`).

Both backends implement the same `Predictor` protocol
(`src/edgeops_ai/prediction.py`), so switching is a config change, not a code
change.

## Training and evaluation pipeline

Everything lives in `scripts/` and runs offline against the collector's
simulation mode:

```powershell
# collect labeled datasets, evaluate the rule baseline, and train a candidate
uv run python -m scripts.run_pipeline --train

# or step by step
uv run python -m scripts.collect_rules_dataset --simulation-seed 42
uv run python -m scripts.evaluate_dataset --input-dir data/generated --output data/generated/rule-baseline-report.json
uv run python -m scripts.train_model
uv run python -m scripts.evaluate_deployed_model --seed-range 62 66
```

Datasets are CSVs under `data/` (gitignored, the JSON reports are kept). Generated
rows carry `dataset_schema_version="2.0"`; stale CSVs are regenerated instead of
being silently reused. Training writes `artifacts/model-candidate.joblib` together
with a training report and refuses to save a model that doesn't pass the quality
gate. Promote the candidate to `artifacts/model.joblib` only after the full
evaluation passes. `evaluate_deployed_model.py` is the honest check: it uses
holdout seeds the model has never seen and goes through the real HTTP endpoint
instead of calling the model directly.

The checked-in production artifact is deliberately still the schema-`1.0`
baseline and is rejected by the schema-`2.0` loader. Keep using the rules backend
until a compatible `lightweight-ml-0.3.0` candidate has passed evaluation and
been promoted.

## Adding another model

Two extension points, depending on what you want:

- **Different estimator, same features:** add it to `CANDIDATE_MODELS` in
  `scripts/train_model.py` and retrain. Anything with `predict()` and
  `predict_proba()` works; the artifact loader and `MlPredictor` don't care what's
  inside.
- **Different prediction method entirely:** implement the `Predictor` protocol,
  add a value to `PredictionBackendName`, and wire it up in the lifespan in
  `main.py`.

## Development

```powershell
uv run pytest
uv run ruff check .
uv run ruff format .
```

Tests cover the API surface, the rule thresholds, feature vectorization and the
model artifact validation.
