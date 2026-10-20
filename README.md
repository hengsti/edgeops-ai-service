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
feature extractor/vectorizer and share dataset provenance in `edgeops_ai.training.common`.
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
across 80 simulated runs (`docs/model-validation/rule-baseline-report.json`) - and more
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
(`docs/model-validation/deployed-ml-report.json`).

Both backends implement the same `Predictor` protocol
(`src/edgeops_ai/prediction.py`), so switching is a config change, not a code
change.

## Training and evaluation

Offline tooling lives in `edgeops_ai.training`. Install development dependencies
with `uv sync --dev`, then use `uv run edgeops-train <command> --help`.
The old `python -m scripts.<name>` commands remain thin compatibility wrappers.
Relative paths are resolved from the terminal's current directory.

| Command | Purpose |
| --- | --- |
| `collect` | Collect one labelled rules dataset |
| `advisor-dataset` | Export labelled InfluxDB aggregate history to Parquet with time-based splits |
| `advisor-labels` | Apply human reviews to an existing Advisor export offline |
| `advisor-health` | Replay device/pipeline advice and a per-device anomaly baseline |
| `pipeline` | Collect missing runs, evaluate the complete matrix, optionally train |
| `train` | Train and evaluate candidates from existing CSVs |
| `evaluate` | Evaluate collected predictions, detection delay, and recovery |
| `evaluate-deployed` | Collect holdout predictions through the real AI API |
| `analyze-features`, `ablate` | Produce feature and ablation evidence |
| `validate` | Check holdout gates and, with `--training-dir`, all release gates |
| `export-onnx` | Convert an existing trusted source artifact without retraining |
| `check-distributions` | Compare normal simulation inputs with saved production observations |

### Advisor dataset

Use the `sensors_1m` and `metrics_5m` history provisioned by
[monitoring #8](https://github.com/hengsti/monitoring/issues/8). Sensor/device and
room tables use one minute; host/container tables preserve five-minute history.
The exporter only reads InfluxDB and writes a new local directory. It never
overwrites an existing dataset. PyArrow is a development dependency; production
inference does not require it.

```powershell
uv sync --dev --locked
$env:INFLUX_URL = 'http://influxdb.smarthome.local'
$env:INFLUX_ORG = 'smarthome'
$env:INFLUX_TOKEN = [IO.File]::ReadAllText('C:\private\influx-token.txt').Trim()
uv run edgeops-train advisor-dataset `
  --start 2026-09-10T00:00:00Z --train-stop 2026-10-01T00:00:00Z `
  --validation-stop 2026-10-10T00:00:00Z --stop 2026-10-19T00:00:00Z `
  --output-dir data/advisor-20261019
```

All boundaries must be explicit UTC and aligned to five minutes. Splits use
`[start, train-stop)`, `[train-stop, validation-stop)`, and
`[validation-stop, stop)` across every entity, without shuffling. Queries run
sequentially in six-hour chunks. The command publishes the directory only after
all queries and files succeed. Rerunning against unchanged aggregate history
with identical options produces the same source and Parquet hashes; late source
writes can change them.

The four long-form tables are `device.parquet`, `host.parquet`,
`container.parquet`, and `room.parquet`. Each row identifies an entity, UTC
window-start time, measurement, field/statistic, dimensions, numeric `value` or
string/boolean `text`, cadence, split, weak labels, and an optional manual label.
Missing fields/windows remain absent. Room means are sample-weighted across
devices with the same historical room tag; min/max/count retain their meanings.
Status rows stay keyed by device ID without assigning a current room to history.
Test devices and known-invalid Docker memory measurements are excluded.
Metrics retain their source tag dimensions, including container state/version.
When a container changes state inside a five-minute window, its partial
aggregates remain distinct under the same entity ID. Use the corresponding
counts when combining their means.

`weak-labels.csv` records counter increases without intervening resets/gaps,
counter resets, unhealthy container status and sensor telemetry gaps of at least
10 missing minutes. All-device gaps use `entity_id=*`. Gaps have no fabricated
measurement rows and do not establish their cause. Dates from #18 are candidate
event references; the unspecified timezone and approximate hours are not ground
truth. Unlabelled observations are not assumed normal. Grafana alerts are not
imported automatically.

`data-card.md` and `data-card.json` describe coverage, missing windows, label
sources, splits and hashes. Review `manual-labels-template.csv` against raw
observations or incident evidence. Fill `label` and `note`, save your review
outside the output directory, then rerun the same command to a new directory
with `--manual-labels <review.csv>`. Blank labels remain unreviewed. Manual labels
stay separate from weak labels; the imported intervals and review notes are
retained in `manual-labels.csv`. Keep the small reviewed set inside validation.
Consumers must build forecast windows and targets inside one split, and fit
preprocessing on training data only. IAQ and quality flags retain source
semantics and need task-specific interpretation.

Apply subsequent human reviews to the existing Parquet export without querying
InfluxDB again:

```powershell
uv run edgeops-train advisor-labels `
  --input-dir data/advisor-issue18-20260910-20261019 `
  --labels data/advisor-reviews-20261019/reviewed-labels.csv `
  --output-dir data/advisor-issue18-reviewed-20261019
```

The review CSV uses `entity_type,entity_id,start,stop,label,note`. Unlike the
exporter's validation-only `--manual-labels`, this explicit review step also
accepts corrections to weak events in the training period. Keep intervals
inside one split and aligned to their entity's cadence. `entity_id=*` applies
to all entities of that type. Contradictory overlapping reviews are rejected;
reimporting the exact same entity/interval replaces its earlier review.

The new dataset preserves measurements, cadence, split assignments and original
weak labels. Consumers use a non-null `manual_label` before `weak_labels`.
Reviews of gaps with no measurement rows remain in `manual-labels.csv`; missing
data is not filled. Existing reviews are retained when applying another file.
Source Parquet hashes are checked, new hashes/counts are recorded, and the
directory is published only after every review matches measurements or a known
gap. Original exports and edited CSVs remain unchanged. Cause and timing details
belong in review notes; diagnosis/recovery windows are not precise fault-onset
measurements. Human reports do not imply independent verification of the cause.

### Advisor health replay

Replay the reviewed #18 export locally. This does not change the incident
classifier, query live services, or deploy an Advisor API (#22).

```powershell
uv run edgeops-train advisor-health `
  --input-dir data/advisor-issue18-reviewed-20261019 `
  --pipeline-csv data/advisor-issue19-pipeline-20261019/pipeline.csv `
  --output-dir data/advisor-health
```

The optional pipeline input is an InfluxDB CSV export of **raw counters**, with
original source tags retained. #18's four Parquet tables do not contain these
Prometheus fields. Export the following query from InfluxDB for the same UTC
interval as the dataset; smaller consecutive ranges can be concatenated,
including their repeated headers. Missing history stays missing.

```flux
from(bucket: "metrics")
  |> range(start: 2026-09-10T00:00:00Z, stop: 2026-10-19T00:00:00Z)
  |> filter(fn: (r) => r._measurement == "prometheus")
  |> filter(fn: (r) => contains(value: r._field, set: [
    "dlq_messages_published_total", "ingest_incoming_invalid_json_total",
    "ingest_validate_raw_failed_total", "emqx_messages_dropped"
  ]))
  |> drop(columns: ["_start", "_stop"])
```

The rules produce evidence and suggested checks, with these initial thresholds:

| Finding | Required evidence |
| --- | --- |
| Missing telemetry | At least 10 missing sensor minutes after a device was first observed; all-device silence supersedes overlapping individual gaps |
| Sensor/delivery errors | Positive counter increases across adjacent minutes, excluding observed resets and uptime decreases; missing counters are not zero |
| Reconnects | At least two observed MQTT connection-counter increments in a daily window |
| Weak Wi-Fi | At least 60 observed RSSI minutes, with at least 80% at or below -75 dBm |
| Suspected memory decline | At least seven distinct days with 60 memory observations each and compatible uptime continuity; daily-minimum trend over at most 30 days is at most -256 bytes/day and falls at least 4,096 fitted bytes |
| IAQ quality | Reported warm-up, or a minimum score at or below 59.8 with a mean at least 70 or uptime below 60 seconds without a warm-up flag; flag for review rather than infer poor room air |
| Pressure offset | At least four simultaneous sensors, so each has three independent peers; at least 60 residuals versus the peers' median, median offset at least 0.8 hPa and median absolute deviation at most 0.25 hPa |
| Pipeline errors | Positive raw-counter differences within two minutes on the same source series; initial totals, negative resets and longer gaps do not count |

Daily summaries stop at train/validation/test boundaries, including intraday
cuts. The replay does not infer the cause of silence, diagnose a leak, or prove
that an IAQ aggregate contains exactly one bad raw sample. Pressure advice asks
for an elevation/placement check before calibration. Gap timestamps indicate
when a running detector could first notice missing data; they do not prove an
alert was delivered while the server was unavailable.

IAQ follows the ingestion service's 0–100 score, where 100 is best and scores at
or below 59.8 map to Very Unhealthy or worse. A poor minimum with a much better
minute mean is a possible transient, not proof of one bad raw sample. Reported
warm-up is retained as a quality flag even when ingest correctly omits the IAQ
score; the replay does not invent a replacement score.

The development-only [IsolationForest baseline](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
uses hourly median RSSI and free memory, requiring at least 45 complete minutes
per hour and 24 training hours per device. Sparse failure counters remain in the
rules rather than being filled for ML. Each forest uses 64 trees, at most 256
samples per tree, seed 42 and one CPU worker. Fitting uses only complete training
windows and excludes reviewed fault windows. Hours crossing a split are skipped.
Unlabelled training history is an unsupervised reference, not confirmed normal.
Scores below `-1e-12` for three consecutive hours produce advice to inspect the
metrics; near-zero floating-point noise is neutral. Forecast residuals and ONNX
delivery belong to #26; this command does not create a deployment artifact.

The new output directory contains `advice.json`, `advice-review.csv`, hourly
`anomaly-scores.csv`, and `report.json` with input hashes, training feature hashes,
model settings, signal coverage and acceptance results. Existing outputs are
never overwritten, source files stay unchanged, and publication is atomic.

Review the advice CSV, filling `verdict` with `true-positive` or `false-positive`
and adding a `note`. Leave uncertain findings blank. Previously reviewed outage
intervals can confirm matching staleness advice; other weak labels do not count
as independent truth. Review every validation advice item for the false-positive
target, and representative historical items for the eight findings in #19.
The named historical cases include the September 11–14 outage, `ad89` sensor
errors, `84be`/`57ad` Wi-Fi and the kitchen pressure offset. A warning on another
subject does not verify those cases.

```powershell
uv run edgeops-train advisor-health `
  --input-dir data/advisor-issue18-reviewed-20261019 `
  --pipeline-csv data/advisor-issue19-pipeline-20261019/pipeline.csv `
  --reviews data/advisor-health/advice-review.csv `
  --output-dir data/advisor-health-reviewed
```

Review IDs bind the full advice evidence and detector to the input file hashes.
Changed inputs require a fresh review. The false-positive rate is the confirmed
false-advice count divided by validation calendar days; it remains unknown while
any validation advice is unreviewed. Signal coverage only means a rule fired.
Historical reproduction also requires a reviewed true positive for the intended
case. Acceptance requires all eight historical findings, a fitted baseline for
each device and fewer than one false advice item per validation day. The command
returns **1** when acceptance is pending or failed, while retaining its reports;
it returns **0** only when acceptance passes.

### Incident classifier workflow

Production polling keeps its default `http://localhost:8095`. Training collection
uses the independent simulation collector at `http://localhost:8097`, with explicit
read/admin credentials. Start that collector first, then run from this checkout:

```powershell
$run = Get-Date -Format 'yyyyMMdd-HHmmss'
$training = "data/training-$run"
$holdout = "data/holdout-$run"

uv run edgeops-train pipeline --seed-range 42 61 --speed 100 --interval 0.05 --train `
  --output-dir $training

$report = Get-Content "$training/training-report.json" -Raw | ConvertFrom-Json
$candidate = $report.artifact_path
$modelVersion = $report.model_version

# Export the fitted candidate; this does not retrain it.
$onnxCandidate = [System.IO.Path]::ChangeExtension($candidate, '.onnx')
$sourceHash = (Get-FileHash $candidate -Algorithm SHA256).Hash
uv run edgeops-train export-onnx --input $candidate --output $onnxCandidate --sha256 $sourceHash

uv run edgeops-train analyze-features --input-dir $training
uv run edgeops-train ablate --input-dir $training
```

Training and pipeline share the default candidate path
`artifacts/model-<contract>-<yyyymmdd>/model-<contract>-<yyyymmdd>.joblib` (UTC date).
A sibling `manifest.json` records the artifact metadata and SHA-256. Use
`train --artifact <path>` or `pipeline --model-artifact <path>` to keep multiple
candidates from one day in separate directories. Retraining the default replaces
that day's candidate; it does not promote it to the runtime default.

Reports default to the selected dataset directory: `rule-report.json`,
`training-report.json`, `feature-audit.json`, and `ablation-report.json`.
The pipeline reuses only complete CSVs with the current schema and matching
simulation manifest, including scenario, seed, severity variant, and run ID.
Otherwise it collects and validates a replacement before publishing it.
Training excludes recovery rows, uses leave-one-seed-out evaluation, requires
every label, and writes the candidate only when its quality gate passes.
Failed stages return a nonzero exit status.

For a release evaluation, start a separate AI instance on port 8096 with the
candidate ONNX file and polling disabled. In a second terminal:

```powershell
$env:PREDICTION_BACKEND = 'ml'
$env:MODEL_ARTIFACT_PATH = '<ONNX candidate path from the training terminal>'
$env:MODEL_ARTIFACT_SHA256 = (Get-FileHash $env:MODEL_ARTIFACT_PATH -Algorithm SHA256).Hash
$env:COLLECTOR_POLLING_ENABLED = 'false'
uv run uvicorn edgeops_ai.main:create_app --factory --port 8096
```

Back in the training terminal, collect unseen holdout seeds and run the combined
release validator:

```powershell
uv run edgeops-train evaluate-deployed --expected-model-version $modelVersion `
  --seed-range 62 71 --speed 50 --interval 0.05 `
  --simulation-base-url http://localhost:8097 --ai-base-url http://localhost:8096 `
  --output-dir $holdout

uv run edgeops-train check-distributions --input-dir $training `
  --production-observations data/production-observations.jsonl

uv run edgeops-train validate --input-dir $holdout --training-dir $training `
  --expected-model-version $modelVersion
```

The deployed evaluation sends observations through `POST /v1/observations`;
it checks the backend/model version and stores `candidate-report.json` alongside
the CSVs and `simulation-manifest.json`. The validator combines holdout checks
(unseen seeds, contract, finite features, version, accuracy, recall, detection,
and recovery) with training, feature availability, ablation, and recovery gates.
Its `release-validation.json` records every gate. Omitting `--training-dir`
runs holdout and production-distribution checks; `--training-seed-range` then
defaults to 42–61.

Release validation also requires `distribution-report.json`. It checks the recorded
feature results, contract version, and source hashes; missing or changed evidence
fails the release. With `--training-dir`, the report defaults to that training
directory; otherwise use `--distribution-report <path>` or place it in the holdout
directory. The legacy acceptance command enforces this gate too.

Save fresh observations from a representative normal production period as JSONL:
one `ObservationV1.model_dump_json()` per line from `CollectorClient.fetch_observation()`.
Keep the real production metadata, device roster, and broker evidence. Use at least
21 observations that produce 20 usable windows after warmup and counter resets.
Then run:

```powershell
uv run edgeops-train check-distributions --input-dir $training `
  --production-observations data/production-observations.jsonl

# Or include the check before training in the pipeline:
uv run edgeops-train pipeline --output-dir $training --seed-range 42 61 --train `
  --production-observations data/production-observations.jsonl
```

The comparison uses stable simulation rows labelled `normal`, excluding recovery.
It checks all 46 vectorized model inputs, including availability, and the unclipped
`processing_ratio` so the old `1.2` mismatch remains visible. Each feature records
both value ranges, the fraction outside the simulation range, and the
[two-sample KS distance](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html).
Defaults fail a feature above distance `0.3` or outside fraction `0.1`;
`check-distributions` exposes both thresholds. These are drift limits, with no
statistical significance claim. Use representative data to assess their suitability.
The report includes source hashes, capture coverage, queue-growth range, and replayed
rules predictions. Unlabelled production observations do not establish accuracy.
Offline training without snapshots remains available for experiments; release
validation requires the production comparison.

The [2026-10-18 production evaluation](docs/model-validation/issue16-production-2026-10-18.md)
confirmed comparable counters and normal rules/ONNX predictions. Its distribution
gate still failed on eight inputs; the saved baseline is not approved for a new
release against that capture.

Generated datasets and candidate reports under `data/` stay outside git.
Obsolete local model candidates are archived under `artifacts/archive/`.
After all release gates pass, promote explicitly: copy the source candidate to
`artifacts/model.joblib` and its ONNX export to `artifacts/model.onnx`, record
the new hashes and validation evidence below, then rebuild/restart with the new
`MODEL_ARTIFACT_SHA256`. The runtime default is always `artifacts/model.onnx`.

## Production counter compatibility

The ingestion counter fix (ingestion-service #15) must be deployed before collecting
new production evidence. `ingest_messages_enqueued_total` and
`ingest_messages_processed_total` both cover sensor and status messages, labelled
by `kind`. The collector sums both sides across the same kinds; the AI uses their
deltas for processing ratio and queue growth. It does not mix the combined enqueue
count with the sensor-only processed count. A ratio above one can still describe
backlog drainage, so the runtime keeps the measured value.

After a successful production ingestion-metrics response, known lazy failure
counters that are absent default to zero. Explicit `null` optional metrics stay
unknown, and simulation masks remain intact. Gauges and other required counters
are not inferred from missing data; in particular, missing InfluxDB health is not
treated as healthy. Failed scrapes continue to fail visibly.

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
to the deployed evaluation and release validator so evaluating the wrong candidate still fails.

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
| `artifacts/model.joblib` | Approved | `bfbffdc925185fb308f8c1073adb96b80a3eaea480e8f5caa3bada549ee360f7` | `lightweight-ml-0.5.0`, decision tree (max depth 7), 46 features, dataset schema 2.2, trained 2026-10-11 on 3174 stable rows from seeds 42-61 (run `data/model-v0.5-20261011-221326`); leave-one-seed-out accuracy 0.977 | `docs/model-validation/deployed-ml-report.json` (seeds 62-71, stable accuracy 0.993); holdout and acceptance gates passed |

When promoting a new candidate, replace the rows above with its version, hashes, training run
directory, seeds, and the holdout/acceptance report that approved it.

Tracked reports are kept as evidence for claims in this README:
`docs/model-validation/rule-baseline-report.json` (rule baseline) and
`docs/model-validation/deployed-ml-report.json` (holdout of the approved v0.5 model). All other
generated data under `data/` is gitignored.

## Adding another model

Two extension points, depending on what you want:

- **Different estimator, same features:** add it to `CANDIDATE_MODELS` in
  `edgeops_ai.training.train` and retrain. The source classifier needs `predict_proba()`,
  `classes_`, and `n_features_in_`; training evaluation also uses `predict()`. Export
  it with `edgeops-train export-onnx` and verify parity before promotion.
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
model artifact validation. An offline pipeline smoke test runs CSV evaluation and real
training on a tiny fixture dataset in under one minute.

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
