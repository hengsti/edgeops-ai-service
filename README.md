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
  "model_version": "lightweight-ml-0.4.0",
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

The old six-feature model classified all 1120 stable-phase samples correctly across
five unseen simulation seeds evaluated through the deployed API, and flagged every
fault at the first sample of the fault phase
(`data/holdout/deployed-ml-report.json`).

Both backends implement the same `Predictor` protocol
(`src/edgeops_ai/prediction.py`), so switching is a config change, not a code
change.

## Training and evaluation pipeline

Everything lives in `scripts/`. Production polling keeps its default
`http://localhost:8095`; dataset generation talks only to the independent
simulation collector on `http://localhost:8097`:

```powershell
# Start the updated simulation collector first (see its README).
# Use a fresh directory: version 0.4 requires regenerated schema-2.1 data.
$run = Get-Date -Format 'yyyyMMdd-HHmmss'
$training = "data/model-v0.4-$run"
$candidate = "artifacts/model-v0.4-$run.joblib"

# collect 20 grouped seeds, evaluate rules, and train a version 0.4 candidate
uv run python -m scripts.run_pipeline --seed-range 42 61 --speed 100 --interval 0.05 --train `
  --output-dir $training `
  --evaluation-report "$training/rule-report.json" `
  --training-report "$training/training-report.json" `
  --model-artifact $candidate

# repeat offline training on the freshly generated dataset
uv run python -m scripts.train_model --input-dir $training `
  --artifact $candidate --report "$training/training-report.json"

# contract-v2 evidence and deterministic acceptance gates
uv run python -m scripts.analyze_features --input-dir $training
uv run python -m scripts.evaluate_feature_ablation --input-dir $training
```

Before promotion, run the selected experimental artifact through the real API on
unused contract-v2 seeds. Start the AI service with `PREDICTION_BACKEND=ml`,
`MODEL_ARTIFACT_PATH=<experimental artifact>`, and collector polling disabled,
then run:

```powershell
# In a separate terminal, from edgeops-ai-service:
$env:PREDICTION_BACKEND = 'ml'
$env:MODEL_ARTIFACT_PATH = '<path printed as $candidate above>'
$env:COLLECTOR_POLLING_ENABLED = 'false'
uv run uvicorn edgeops_ai.main:create_app --factory --port 8096

# In the training terminal:
uv run python -m scripts.evaluate_deployed_model `
  --seed-range 62 71 `
  --speed 50 `
  --interval 0.05 `
  --simulation-base-url http://localhost:8097 `
  --ai-base-url http://localhost:8096 `
  --output-dir "data/holdout-v0.4-$run" `
  --report "data/holdout-v0.4-$run/candidate-report.json"

uv run python -m scripts.validate_candidate_holdout --input-dir "data/holdout-v0.4-$run"
uv run python -m scripts.validate_phase6_acceptance `
  --input-dir $training `
  --holdout-report "data/holdout-v0.4-$run/candidate-report.json"
```

The holdout command sends explicit balanced variants, uses separate simulation
read/admin keys, and writes a contract-v2 `simulation-manifest.json` beside the
CSVs. It does not load the model directly; predictions must come from the running
FastAPI ML backend.

Datasets are CSVs under `data/` (gitignored, the JSON reports are kept). Generated
rows carry `dataset_schema_version="2.1"`; stale CSVs are regenerated instead of
being silently reused. Reuse also requires a matching contract-v2
`simulation-manifest.json`, including the planned severity variant, resolved effects,
masked optional groups, collector version, and exact CSV run ID. The pipeline passes
separate simulation read/admin keys to child processes and never relies on the
production `.env`. Training writes only the explicitly selected experimental path
together with a training report and refuses to save a model that doesn't pass the
quality gate. Promote a candidate to `artifacts/model.joblib` only after the full
evaluation passes. `evaluate_deployed_model.py` is the honest check: it uses
holdout seeds the model has never seen and goes through the real HTTP endpoint
instead of calling the model directly.

Version 0.4 fixes two training-data errors: healthy simulated devices previously had
exactly zero heartbeat age on every read, which let the tree learn a false heartbeat
cutoff below the normal two-second publishing interval; accelerated counters also used
wall-clock timestamps, inflating rate features by the simulation speed. The updated
simulation uses periodic heartbeat ages and timestamps in simulated seconds. Its profile's
`heartbeat_interval_seconds` defaults to `2.0`, matching the ESP32 firmware; adjust it to the
device cadence before collecting data for other hardware. Restart or rebuild the simulation
collector before generating new data. Old schema-2.0 CSVs must be regenerated, not relabeled.

Training, loading, and deployed evaluation use `lightweight-ml-0.4.0`. Existing 0.3 artifacts
need retraining; changing their version string does not fix their learned decisions.
The checked-in `artifacts/model.joblib` is still the old six-feature baseline, so using
the ML backend with that file fails validation. Load the candidate explicitly for evaluation.
After the training and holdout gates pass, promote the selected artifact with
`Copy-Item $candidate artifacts/model.joblib -Force`, rebuild the AI image (its Dockerfile
copies that file), and restart the AI service. Use the rules backend until promotion.

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
