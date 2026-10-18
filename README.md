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
 |   Analyzer                                     |
 |      |                                         |
 |      v                                         |
 |   FeatureExtractor (rates + ratios)            |
 |          |                                     |
 |          v                                     |
 |   Predictor:  rules  -or-  ml                  |
 +------------------------------------------------+
        |
        v
 GET /v1/detections/latest
```

Observations reach the service either through the built-in collector polling loop
or by pushing them to `POST /v1/observations` with `COLLECTOR_POLLING_ENABLED=false`.
While polling is enabled, pushes return HTTP 409 so the two sources cannot mix
observation windows. Analyzer updates are serialized with an async lock on the
event loop. The feature extractor needs two consecutive snapshots to compute
rates, so the first response after startup is always `"status": "warming-up"`.

The runtime has explicit ownership boundaries:

| Module | Responsibility |
| --- | --- |
| `main.py` | Select the predictor, assemble resources, and close the polling task and HTTP client |
| `api.py` | HTTP routes and response delivery |
| `analysis.py` | One observation window and latest analysis, shared by polling and HTTP submission |
| `polling.py` | Fetch observations, retain the latest result during errors, and resume on recovery |
| `collector_client.py` | Authenticated collector requests and conversion into observations |
| `features.py`, `feature_groups.py` | Observation windows, grouped calculations, and stream/counter reset handling |
| `schemas.py` | Observation schemas and the metric catalog used by the client and counter reset checks |
| `rules.py`, `ml.py` | Prediction, with the shared ML feature contract and artifact validation in `ml.py` |

Rules mode loads no ML modules. ML mode loads the same approved artifact and still
requires its SHA-256 before deserialization. The training scripts reuse the runtime
feature extractor/vectorizer and share dataset provenance in `scripts/common.py`.
`MODEL_FEATURES` in `ml.py` defines the ordered model columns and their calculations.
Training parses scalar feature fields from `FeatureVector` and uses the same vectorizer.
Availability comes from missing values; unknown telemetry stays distinct from a measured zero.

In both production and simulation, `persistence_ratio` is `1.0` when InfluxDB is healthy
and the observation window contains no write errors: asynchronous WAL batches can cross
window boundaries. Otherwise it is the written-line/transform-success ratio, bounded to
`[0, 1]`. Regenerate older simulation CSVs before future retraining to use this shared rule.
See the [refactoring record](docs/refactoring-2026-10-12.md) for scope and validation.

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
`artifacts/model.onnx`; to run the ML backend, pass its SHA-256 (see
[Model artifacts](#model-artifacts-and-reports)):

```powershell
docker run --rm -p 8080:8080 -e PREDICTION_BACKEND=ml `
  -e "MODEL_ARTIFACT_SHA256=$((Get-FileHash artifacts/model.onnx -Algorithm SHA256).Hash)" `
  edgeops-ai-service
```

To test another artifact, mount it over the bundled one with
`-v <path>:/app/artifacts/model.onnx:ro` and pass that file's SHA-256.

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
| `MODEL_ARTIFACT_PATH` | `artifacts/model.onnx` | trained model, only used by the ml backend |
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
| `GET` | `/health` | liveness, polling status, active backend and model version |
| `GET` | `/ready` | readiness (503 before the first successful poll or during polling failures) |
| `GET` | `/metrics` | Prometheus text metrics for polling failures and prediction freshness |
| `POST` | `/v1/observations` | analyze one observation (409 while collector polling is enabled) |
| `GET` | `/v1/detections/latest` | most recent analysis with freshness (404 before the first one) |

`/health` stays HTTP 200 for liveness, but reports `status: degraded` after the
first failed collector poll. It includes `last_poll_success_at` (UTC, or `null`
before any successful poll), `last_error`, and `consecutive_failures`. The next
successful poll restores `ok`, clears the error, and resets the failure count.
With polling disabled, these fields remain `null`, `null`, and `0`.

`/ready` returns HTTP 200 with `status: ready` after a successful poll, and HTTP 503
with `status: not-ready` before the first success or after a failed poll. It becomes
ready again on recovery. Push-only mode is ready immediately.

`/metrics` serves Prometheus text format 0.0.4 without labels:

- `edgeops_poll_failures_total` (counter): all failed collector polls since startup;
  recovery does not reset it.
- `edgeops_last_prediction_timestamp` (gauge): Unix timestamp of the observation
  used by the last completed prediction, or `0` before the first prediction.
  Failed polls and warm-up observations retain the last prediction timestamp.

The latest detection adds `observed_at` (the observation's `captured_at`),
`age_seconds` (calculated when requested, clamped to zero for future timestamps),
and `stale`. A detection is stale after a failed poll or when its age exceeds
twice `COLLECTOR_POLL_INTERVAL_SECONDS` (20 seconds by default, also in push mode).
The last result remains available during outages. These freshness fields are
specific to `GET /v1/detections/latest`.

The latest detection response looks like this:

```json
{
  "status": "predicted",
  "captured_at": "2024-01-01T02:03:03+01:00",
  "observed_at": "2024-01-01T02:03:03+01:00",
  "age_seconds": 0.4,
  "stale": false,
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

# collect 20 grouped seeds, evaluate rules, and train a candidate
uv run python -m scripts.run_pipeline --seed-range 42 61 --speed 100 --interval 0.05 --train `
  --output-dir $training `
  --evaluation-report "$training/rule-report.json" `
  --training-report "$training/training-report.json" `
  --model-artifact $candidate

# repeat offline training on the freshly generated dataset
uv run python -m scripts.train_model --input-dir $training `
  --artifact $candidate --report "$training/training-report.json"
$modelVersion = (Get-Content "$training/training-report.json" -Raw | ConvertFrom-Json).model_version

# Export the fitted candidate for serving; this does not retrain it.
$onnxCandidate = [System.IO.Path]::ChangeExtension($candidate, '.onnx')
$sourceHash = (Get-FileHash $candidate -Algorithm SHA256).Hash
uv run python -m scripts.export_onnx --input $candidate --output $onnxCandidate --sha256 $sourceHash

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
$env:MODEL_ARTIFACT_PATH = '<ONNX path printed by export_onnx above>'
$env:MODEL_ARTIFACT_SHA256 = (Get-FileHash $env:MODEL_ARTIFACT_PATH -Algorithm SHA256).Hash
$env:COLLECTOR_POLLING_ENABLED = 'false'
uv run uvicorn edgeops_ai.main:create_app --factory --port 8096

# Or the same in Docker, publishing container port 8080 on host port 8096:
docker run --rm -p 8096:8080 -e PREDICTION_BACKEND=ml -e COLLECTOR_POLLING_ENABLED=false `
  -e MODEL_ARTIFACT_SHA256=$env:MODEL_ARTIFACT_SHA256 `
  -v "${PWD}/$($env:MODEL_ARTIFACT_PATH):/app/artifacts/model.onnx:ro" edgeops-ai-service

# In the training terminal (the script runs on the host, so it uses host ports):
uv run python -m scripts.evaluate_deployed_model `
  --expected-model-version $modelVersion `
  --seed-range 62 71 `
  --speed 50 `
  --interval 0.05 `
  --simulation-base-url http://localhost:8097 `
  --ai-base-url http://localhost:8096 `
  --output-dir "data/holdout-v0.5-$run" `
  --report "data/holdout-v0.5-$run/candidate-report.json"

uv run python -m scripts.validate_candidate_holdout --input-dir "data/holdout-v0.5-$run" `
  --expected-model-version $modelVersion
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
quality gate. Promote a candidate's source and ONNX artifacts only after the full
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

The approved model is `lightweight-ml-0.5.0`. New training runs record a supplied or
generated model version independently of the service release. Version 0.5 adds the
`mqtt-broker-unavailable` label and two broker columns (`broker_unavailable`,
`broker_metrics_available`), so 0.4 and older artifacts need retraining on schema-2.2 data;
changing their version string does not fix their learned decisions. Training requires every
label; the simulation collector's `mqtt-broker-unavailable` scenario provides it (EMQX unhealthy,
restarting or exited, stale heartbeats and dropping traffic, then recovery), while the other
scenarios report a healthy broker. Rebuild the simulation collector before collecting 0.5 data.
The checked-in `artifacts/model.joblib` is the promoted 0.5 source model;
`artifacts/model.onnx` is its exported serving artifact. Load new candidates explicitly
for evaluation. After the training and holdout gates pass, copy `$candidate` to
`artifacts/model.joblib` and `$onnxCandidate` to `artifacts/model.onnx`, record both in
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

Production serving uses ONNX Runtime and does not import joblib or scikit-learn.
The image contains only the approved ONNX artifact. Its SHA-256 must match
`MODEL_ARTIFACT_SHA256` before an inference session is opened.

The development environment retains joblib loading for trusted source artifacts and
conversion. Loading a joblib pickle can execute code, so the export command requires
the recorded source SHA-256 and verifies it before loading.

Settings validate the hash before ML startup: it must contain 64 hexadecimal digits.
Rules mode accepts an unset or blank hash.

Each artifact manifest records its own non-empty `model_version` and
`feature_contract_version`. The current feature contract is `1.0`; loading also checks
the exact 46-column order, estimator feature count, known unique classes, and matching
manifest labels. Inference uses one `predict_proba()` call and selects its highest
probability. A new compatible model version needs no service code change.

Training accepts `--model-version <name>` or generates a UTC timestamp version when
omitted. The training report records that version; pass it as `--expected-model-version`
to both holdout commands so evaluating the wrong candidate still fails.

The approved ONNX artifact was exported directly from the fitted v0.5 tree without
retraining. It retains model version `lightweight-ml-0.5.0`, feature contract `1.0`,
and all 46 inputs. Its embedded manifest records labels in probability-column order.
Tests compare predictions and probabilities against the source model within float32
tolerance. Conversion does not need training data or change the source file.

For the converted deployment, set `MODEL_ARTIFACT_PATH=artifacts/model.onnx` and use the
ONNX SHA-256 below. Older artifacts without a contract version are rejected; new
training runs write the required manifest.

The source `artifacts/model.joblib` and serving `artifacts/model.onnx` are tracked.
Experimental and candidate
artifacts stay local (`artifacts/*` is gitignored); delete or archive them outside
git once a candidate is promoted or rejected.

| Artifact | Status | SHA-256 | Provenance | Validation |
| --- | --- | --- | --- | --- |
| `artifacts/model.onnx` | Approved conversion | `3dcef00469e3a1157d21ccaaf3315dee45c06f1a56be6668fec31e5e280cb4e8` | Direct export of the source below; same model version and feature contract; no retraining | `tests/test_onnx.py`: prediction and probability parity; production host validation pending |
| `artifacts/model.joblib` | Approved | `bfbffdc925185fb308f8c1073adb96b80a3eaea480e8f5caa3bada549ee360f7` | `lightweight-ml-0.5.0`, decision tree (max depth 7), 46 features, dataset schema 2.2, trained 2026-10-11 on 3174 stable rows from seeds 42-61 (run `data/model-v0.5-20261011-221326`); leave-one-seed-out accuracy 0.977 | `data/holdout/deployed-ml-report.json` (seeds 62-71, stable accuracy 0.993); holdout and acceptance gates passed |

When promoting a new candidate, replace the rows above with its version, hashes, training run
directory, seeds, and the holdout/acceptance report that approved it.

Tracked reports are kept as evidence for claims in this README:
`data/generated/rule-baseline-report.json` (rule baseline) and
`data/holdout/deployed-ml-report.json` (holdout of the approved v0.5 model). All other
generated data under `data/` is gitignored.

## Adding another model

Two extension points, depending on what you want:

- **Different estimator, same features:** add it to `CANDIDATE_MODELS` in
  `scripts/train_model.py` and retrain. The source classifier needs `predict_proba()`,
  `classes_`, and `n_features_in_`; training evaluation also uses `predict()`. Export
  it with `scripts.export_onnx` and verify parity before promotion.
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
