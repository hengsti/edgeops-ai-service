# edgeops-ai-service

Anomaly detection for the EdgeOps smart-home stack. The service watches pipeline
metrics coming from the edge collector, classifies the current state as normal or
one of five fault types, and tells you what to check first.

Fault labels it knows about:

- `ingestion-backpressure` – the queue grows faster than messages get processed
- `missing-device-heartbeat` – a device went silent
- `malformed-sensor-payload` – payloads fail transformation and land in the DLQ
- `influxdb-write-failure` – data gets transformed but never persisted
- `mqtt-broker-unavailable` – the EMQX broker is confirmed down, so devices look silent

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
uv run uvicorn edgeops_ai.main:create_app --factory --reload --port 8080
```

Or with Docker:

```powershell
docker build -t edgeops-ai-service .
docker run --rm -p 8080:8080 edgeops-ai-service
```

The quickstart uses the default `rules` backend. The image bundles the approved
`artifacts/model.joblib`; to run the ML backend, pass its SHA-256 (see
[Model artifacts](#model-artifacts-and-reports)):

```powershell
docker run --rm -p 8080:8080 -e PREDICTION_BACKEND=ml `
  -e "MODEL_ARTIFACT_SHA256=$((Get-FileHash artifacts/model.joblib -Algorithm SHA256).Hash)" `
  edgeops-ai-service
```

To test another artifact, mount it over the bundled one with
`-v <path>:/app/artifacts/model.joblib:ro` and pass that file's SHA-256.

### Ports

The container always listens on port `8080`; `-p HOST:CONTAINER` publishes it on a
host port. Which URL to use depends on where the caller runs:

| Caller | AI service URL |
| --- | --- |
| Host (browser, scripts, `uv run` tools) | `http://localhost:<host port>`, e.g. `http://localhost:8080` for `-p 8080:8080` |
| Another container on the same Docker network | `http://<container name>:8080` (container port, no `-p` needed) |

The same applies in reverse to `COLLECTOR_BASE_URL`: inside the container,
`localhost` is the container itself. Point it at a collector container on the same
network (`http://<collector container>:<container port>`) or, for a collector on the
host, `http://host.docker.internal:8095` (Docker Desktop; on Linux add
`--add-host=host.docker.internal:host-gateway`).

Configuration is plain environment variables. Copy `.env.example` to `.env` (never
commit it; `COLLECTOR_API_KEY` is a secret):

| Variable | Default | Purpose |
| --- | --- | --- |
| `PREDICTION_BACKEND` | `rules` | `rules` or `ml` |
| `MODEL_ARTIFACT_PATH` | `artifacts/model.joblib` | trained model, only used by the ml backend |
| `MODEL_ARTIFACT_SHA256` | *(empty)* | required for the ml backend; the artifact is refused unless its SHA-256 matches |
| `COLLECTOR_BASE_URL` | `http://localhost:8095` (Compose: `http://edgeops-collector:8095`) | collector metrics API |
| `COLLECTOR_API_KEY` | `local-dev-key` | sent as `X-EdgeOps-Key` |
| `COLLECTOR_POLLING_ENABLED` | `true` | disable if you only push observations |
| `COLLECTOR_POLL_INTERVAL_SECONDS` | `10` | polling interval |
| `COLLECTOR_TIMEOUT_SECONDS` | `3` | request timeout |
| `COLLECTOR_DEVICE_IDS` | *(empty)* | comma-separated devices for heartbeat checks |

### Docker Compose

`compose.yaml` defines the service as this repo owns it: build from `.`, `expose: 8080`
(no host port), the optional `.env`, and the image `HEALTHCHECK` on `/health`.
`COLLECTOR_BASE_URL` defaults to `http://edgeops-collector:8095`; a value in the shell
or in this repo's `.env` replaces it, so leave it unset there unless you need another
collector.

Standalone:

```powershell
docker compose up -d --build
docker compose ps    # edgeops-ai-service becomes healthy
```

Without the collector on the same network, polling only logs errors. To reach the
service from the host, add a local `compose.override.yaml` (merged automatically):

```yaml
services:
  edgeops-ai-service:
    ports: ["8096:8080"]
```

In the Server stack, this repo is checked out at `./edgeops-ai/service` and included
together with a stack-side override that adds the `8096:8080` publish and
`depends_on: edgeops-collector`:

```yaml
include:
  - path:
      - ./edgeops-ai/service/compose.yaml
      - ./edgeops-ai-service.override.yaml
    env_file: ./.env
```

Inside the stack network, consumers such as control-ui use
`http://edgeops-ai-service:8080`.

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
  "model_version": "lightweight-ml-0.5.0",
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
regression and a depth-7 decision tree; depth 6 lost ingestion-backpressure recall once the
broker label was added); the tree won. Features are
workload-independent ratios rather than raw counters, so the model doesn't just
memorize the simulation's message rates.

The deployed `lightweight-ml-0.5.0` model classified 3782 of 3810 stable-phase samples
(99.3 %) correctly across ten unseen simulation seeds (62-71) evaluated through the
deployed API, and detected all 50 incidents with a mean delay of 2.1 s
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
# Use a fresh directory: version 0.5 requires regenerated schema-2.2 data.
$run = Get-Date -Format 'yyyyMMdd-HHmmss'
$training = "data/model-v0.5-$run"
$candidate = "artifacts/model-v0.5-$run.joblib"

# collect 20 grouped seeds, evaluate rules, and train a version 0.5 candidate
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
$env:MODEL_ARTIFACT_SHA256 = (Get-FileHash $env:MODEL_ARTIFACT_PATH -Algorithm SHA256).Hash
$env:COLLECTOR_POLLING_ENABLED = 'false'
uv run uvicorn edgeops_ai.main:create_app --factory --port 8096

# Or the same in Docker, publishing container port 8080 on host port 8096:
docker run --rm -p 8096:8080 -e PREDICTION_BACKEND=ml -e COLLECTOR_POLLING_ENABLED=false `
  -e MODEL_ARTIFACT_SHA256=$env:MODEL_ARTIFACT_SHA256 `
  -v "${PWD}/$($env:MODEL_ARTIFACT_PATH):/app/artifacts/model.joblib:ro" edgeops-ai-service

# In the training terminal (the script runs on the host, so it uses host ports):
uv run python -m scripts.evaluate_deployed_model `
  --seed-range 62 71 `
  --speed 50 `
  --interval 0.05 `
  --simulation-base-url http://localhost:8097 `
  --ai-base-url http://localhost:8096 `
  --output-dir "data/holdout-v0.5-$run" `
  --report "data/holdout-v0.5-$run/candidate-report.json"

uv run python -m scripts.validate_candidate_holdout --input-dir "data/holdout-v0.5-$run"
uv run python -m scripts.validate_phase6_acceptance `
  --input-dir $training `
  --holdout-report "data/holdout-v0.5-$run/candidate-report.json"
```

The holdout command sends explicit balanced variants, uses separate simulation
read/admin keys, and writes a contract-v2 `simulation-manifest.json` beside the
CSVs. It does not load the model directly; predictions must come from the running
FastAPI ML backend.

Datasets are CSVs under `data/` (gitignored, the JSON reports are kept). Generated
rows carry `dataset_schema_version="2.2"`; stale CSVs are regenerated instead of
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

Training, loading, and deployed evaluation use `lightweight-ml-0.5.0`. Version 0.5 adds the
`mqtt-broker-unavailable` label and two broker columns (`broker_unavailable`,
`broker_metrics_available`), so 0.4 and older artifacts need retraining on schema-2.2 data;
changing their version string does not fix their learned decisions. Training requires every
label; the simulation collector's `mqtt-broker-unavailable` scenario provides it (EMQX unhealthy,
restarting or exited, stale heartbeats and dropping traffic, then recovery), while the other
scenarios report a healthy broker. Rebuild the simulation collector before collecting 0.5 data.
The checked-in `artifacts/model.joblib` is the promoted 0.5 model. Load new candidates
explicitly for evaluation. After the training and holdout gates pass, promote the selected
artifact with `Copy-Item $candidate artifacts/model.joblib -Force`, record it in
[Model artifacts and reports](#model-artifacts-and-reports), and restart the AI service with
the new `MODEL_ARTIFACT_SHA256` (mount the file into the container as shown in the
quickstart).

## Broker versus device diagnosis

A stopped EMQX broker makes every device heartbeat stale, so the service reads the broker
state from the collector (`GET /v1/services/emqx`) on every poll and stores it as
`broker_available` in the observation and the features:

| Collector response | `broker_available` |
| --- | --- |
| `status=running`, `health=healthy` | `true` |
| `status` `exited`, `dead`, `restarting`, `paused`, `created`, `removing`, or `health=unhealthy` | `false` (confirmed outage) |
| `status=unavailable` (Docker inspection failed), `health=starting` or missing, HTTP error, timeout | `null` (unknown) |

Failed broker HTTP requests are retried once within the same poll. If both attempts fail,
the service logs the request error and reports `null`; it does not reuse a previous broker
state. A retry can add up to one additional collector request timeout to the poll.

Pushed observations without the field are accepted and treated as unknown. Only a confirmed
outage produces `mqtt-broker-unavailable`, and it takes precedence over
`missing-device-heartbeat`. Unknown broker telemetry and zero message traffic never do: the
analysis continues with the remaining signals. The health status comes from the EMQX
healthcheck (`emqx ctl status`) in the Server stack; there is no separate MQTT reachability
probe. After the broker comes back, the first observation with restarted counters returns
`warming-up` and the next one is classified normally.

To validate against a real broker, use a test or staging stack, never production:

1. Run the AI service with the rules backend and collector polling, and wait for `normal`.
2. `docker stop emqx`, wait two poll intervals, and confirm `GET /v1/detections/latest`
   reports `mqtt-broker-unavailable` with the EMQX recommendation.
3. `docker start emqx`, wait for the container to report `healthy` and the devices to
   reconnect, and confirm the detection returns to `normal`.

Record the outcome separately from the simulation reports.

## Model artifacts and reports

joblib files are pickles: loading one can run arbitrary code. The ML backend therefore
only loads an artifact whose SHA-256 matches `MODEL_ARTIFACT_SHA256`, and the hash is
checked before `joblib.load`. Only use hashes of artifacts you trained yourself with
`scripts/train_model.py` and recorded below.

`artifacts/model.joblib` is the only tracked artifact. Experimental and candidate
artifacts stay local (`artifacts/*` is gitignored); delete or archive them outside
git once a candidate is promoted or rejected.

| Artifact | Status | SHA-256 | Provenance | Validation |
| --- | --- | --- | --- | --- |
| `artifacts/model.joblib` | Approved | `6112373eb459e3c824f562afef96c2fdff51cf151cb319eb0033be41a3db515a` | `lightweight-ml-0.5.0`, decision tree (max depth 7), 46 features, dataset schema 2.2, trained 2026-10-11 on 3174 stable rows from seeds 42-61 (run `data/model-v0.5-20261011-221326`); leave-one-seed-out accuracy 0.977 | `data/holdout/deployed-ml-report.json` (seeds 62-71, stable accuracy 0.993); holdout and acceptance gates passed |

When promoting a new candidate, replace the row above with its version, hash, training run
directory, seeds, and the holdout/acceptance report that approved it.

Tracked reports are kept as evidence for claims in this README:
`data/generated/rule-baseline-report.json` (rule baseline) and
`data/holdout/deployed-ml-report.json` (holdout of the approved v0.5 model). All other
generated data under `data/` is gitignored.

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
uv run mypy src
```

Tests cover the API surface, the rule thresholds, feature vectorization and the
model artifact validation.

CI (`.github/workflows/ci.yml`) runs on every push to `main`, every pull request and
every `v*` tag. It installs the locked dependencies (`uv sync --locked`), runs ruff,
mypy and pytest, then builds the image and checks that `/health` reports the
`rules` backend.

## Release

Only revisions with a green CI run on `main` get released:

1. Bump `version` in `pyproject.toml` and `__version__` in
   `src/edgeops_ai/__init__.py`, run `uv lock`, merge to `main` and wait for CI.
2. Tag that commit and push the tag (CI runs again on the tag):

   ```powershell
   git tag v0.2.0 <green main commit>
   git push origin v0.2.0
   ```

3. Build the release image from the tag:

   ```powershell
   git checkout v0.2.0
   docker build -t edgeops-ai-service:0.2.0 .
   ```

Images are not published to a registry. A model artifact is not part of a release;
mount it at runtime as shown in the quickstart.

## License

[MIT](LICENSE)
