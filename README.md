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
uv sync --locked --no-dev
uv run --no-sync uvicorn edgeops_ai.main:create_app --factory --reload --port 8080
```

Choose the installation for the workload:

| Workload | Install command | Dependencies |
| --- | --- | --- |
| Rules-only service | `uv sync --locked --no-dev` | HTTP service only; no NumPy, ONNX Runtime, scikit-learn or joblib |
| ML service with the approved ONNX artifact | `uv sync --locked --no-dev --extra ml` | HTTP service plus ONNX Runtime and its dependencies |
| Development, training and evaluation | `uv sync --locked` | Both backends plus the existing development/training tools, including scikit-learn and joblib |

For pip installations, the corresponding runtime choices are `pip install .` and
`pip install '.[ml]'`. Legacy `.joblib` artifacts require the development/training
installation. The optional Advisor training group still uses
`uv sync --locked --group advisor`.

After preparing a runtime environment, use `uv run --no-sync` to start it as shown
above. Plain `uv run` installs the default development group. To select local ML
inference, also set `PREDICTION_BACKEND=ml` and `MODEL_ARTIFACT_SHA256` to the
approved artifact's hash; the existing hash and model-contract validation applies.

Measured on Windows x86_64 with Python 3.13.5 and this lockfile on 2026-10-21,
non-editable installations contain **11.53 MiB** (20 packages) for rules-only and
**94.87 MiB** (25 packages) with `--extra ml`: **83.34 MiB (87.9%) less** for rules.
These are summed file sizes under `site-packages`, excluding bytecode, the Python
interpreter and uv caches. They measure installed packages rather than image size
or runtime memory. Both environments omit scikit-learn and joblib; training keeps them.

Or with Docker:

```powershell
docker build -t edgeops-ai-service .
docker run --rm -p 8080:8080 edgeops-ai-service
```

The quickstart uses the default `rules` backend. Docker explicitly installs the
`ml` extra so the same image supports both backends. It bundles the approved
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
| `GET` | `/v1/state-report` | versioned Advisor findings and coverage (503 when unconfigured/unavailable) |

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

### Advisor state report (schema 1.0)

The Advisor stays in this service, separate from the incident classifier. Enable
it with `ADVISOR_INFLUX_URL`, `ADVISOR_INFLUX_ORG` (default `smarthome`) and
`ADVISOR_INFLUX_TOKEN`. Use a read-only token for `sensors_1m` and `metrics`, kept
in the local environment or `.env`; do not commit it. URL and token must be set
together. With both omitted, the endpoint returns 503 and existing incident
endpoints retain their normal behavior.

`GET /v1/state-report` runs existing device-health, raw pipeline-counter and
indoor-climate rules on the latest bounded 24-hour window. The window ends one
completed minute behind the current UTC minute to allow aggregate task delay.
Queries select only required sensor fields and pipeline counters, preserve
historical device/room and pipeline-series identity, and exclude test devices.
Each response is limited to 32 MiB and HTTP operations have a 10-second timeout.
Malformed, conflicting or nonfinite observations fail the refresh.

Reports are cached for `ADVISOR_REFRESH_SECONDS` (default 300, range 60–3600).
Concurrent requests share one refresh, and parsing/detection runs in worker
threads. Failed refreshes return 503 with a generic error, retain internal finding
history, and retry after 60 seconds; expired reports are not served as current.
The application owns and closes its Influx client. No background training or
additional runtime dependency is introduced.

The response has these fields; OpenAPI documents the exact `StateReport`,
`Finding` and `Evidence` schemas at `/openapi.json`:

| Field | Meaning |
| --- | --- |
| `schema_version` | `1.0`, independent of service/model versions; breaking changes require a new schema version |
| `generated_at` | UTC report-generation time, unchanged while the response is cached |
| `observed_at`, `window_start` | Exclusive observation-window end and inclusive start |
| `status` | `complete` or `partial`, according to category coverage |
| `coverage` | Per-category `available`, `no-data` or `deferred`; availability means source observations exist, not that every possible detector was evaluated |
| `overall` | `critical` for critical findings; `attention` for warnings or incomplete coverage; otherwise `ok` |
| `findings` | Deduplicated findings, sorted by stable ID |

Each finding includes `id`, `category` (`device`, `pipeline`, `climate`, `host`,
`maintenance`), `severity` (`info`, `warning`, `critical`), `subject`, `summary`,
`recommendation`, `detector`, `first_seen`, `last_seen`, `window_start` and
`window_stop`. Its evidence entries contain `metric`, `window` (duration in
seconds, for example `7200s`), numeric/string `value`, and optional `baseline`.
Missing baselines remain null. `confidence` is a nullable number from 0 to 1;
the rules return null because their confidence is not calibrated.

IDs identify the finding code, source subject and detector version. Repeated
findings retain `first_seen`; a finding disappears only after two successful
refreshes without it. Failures do not count as recovery. Climate rules retain
their existing threshold hysteresis and hourly room cooldown. Findings summarize
events in the observation window; absence from a later window does not prove
their physical cause was resolved. Finding history is per process and resets
on restart; durable notification delivery is outside this API.

`Detector.detect(AdvisorWindow)` returns existing `Advice` facts; `InfluxReader`
provides the input window. `create_app(advisor_reader=..., advisor_detectors=...)`
accepts fixture readers and additional detector implementations without a plugin
registry or changing the incident predictor. Injected readers remain owned by
their caller. Deterministic detector output is normalized and validated against
the public schema before publishing it.

The initial live reader does not evaluate multi-day memory decline, offline
IsolationForest baselines, host/maintenance ETAs or trained forecasts. Host and
maintenance coverage is explicitly `deferred`, so this initial report is partial
and never silently reports overall `ok` for those missing checks. Remaining work:

- [#31](https://github.com/hengsti/edgeops-ai-service/issues/31): bounded longer-history and host/maintenance runtime adapters, reusing #19/#21 without importing training tools into request handling.
- [#32](https://github.com/hengsti/edgeops-ai-service/issues/32): live model-consumer wiring after #27/#28/#30 and their #26 artifact/contract prerequisites.
- [#29](https://github.com/hengsti/edgeops-ai-service/issues/29): outdoor weather integration.

Grafana JSON consumers can poll this GET endpoint and inspect `status`/`coverage`
alongside findings. No Grafana provisioning, notifications or LLM execution is
included. Optional explainers run off the Pi and consume this JSON; the service
does not call an LLM.

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
| `advisor-climate` | Replay room climate advice and optional TTM humidity/temperature forecasts |
| `advisor-maintenance` | Estimate maintenance thresholds; optionally assess observed host workloads |
| `advisor-forecast` | Fine-tune TTM on CPU, export ONNX and validate native/ONNX parity |
| `advisor-benchmark` | Measure ONNX latency/process memory on native-reference held-out inputs |
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
metrics; near-zero floating-point noise is neutral. The optional TTM residual
detector below consumes #26's ONNX artifact; this command does not export a model.

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

#### TTM forecast-residual health replay (#27)

The intended product is an Advisor that assesses the flat's observed state and
suggests useful actions. TTM supplies expected temperature/humidity patterns from
recent room history; its residual detector compares those expectations with
arriving measurements. Existing climate/device/pipeline rules supply explicit
conditions and recommendations. TTM itself returns numeric forecasts, not a
diagnosis, an explanation of causes or natural-language advice.

The user chose to retain the existing #27 candidate
`advisor-ttm-climate-20261020` without further fine-tuning for this role. Forecast
MAE remains diagnostic evidence; it does not by itself establish or reject
usefulness for anomaly advice. Acceptance still requires useful warnings,
preserved known findings and the reviewed false-positive limit. This choice is
not a verdict on individual review rows or production approval. The current
TTM input is temperature/humidity with 512 recent minutes per device/room; the
live state-report API currently uses deterministic detectors. TTM live wiring
remains #32, and wider historical/resource coverage remains #31.

Use the validated #26 **climate** candidate with its exact chronological dataset:

```powershell
uv run edgeops-train advisor-health `
  --input-dir data/advisor-issue26-climate-20261010-20261019 `
  --forecast-candidate artifacts/advisor-ttm-climate-20261020 `
  --output-dir data/advisor-issue27-health
```

The candidate directory must contain `model.onnx`, `model.onnx.sha256`,
`manifest.json` and the passing `validation-report.json` from #26. The shared
forecast loader validates the SHA-256, embedded/JSON manifests, tensor contract,
parity-report identity and dataset hashes. A different time range or split is
rejected: a model trained in October cannot evaluate September as held-out data.
This offline experiment accepts a validated candidate whose deployment approval
is still false; it does not promote or deploy it. No training framework is used.

The shared #26 window loader provides context, forecast targets and the observed
values in the same channel order. It rejects malformed inputs and skips windows
with gaps, warm-up failures, resets, room changes or split crossings, without
imputation. Every device/room must have complete windows in each split. Forecasts
are compared only to their future targets, starting one minute after the last
input. The pinned real model uses 512 context minutes and 96 target minutes;
successive target blocks do not overlap.

Before evaluating held-out windows, freeze each device/room/horizon/channel scale
at the **99th percentile of absolute training residuals**, floored at `0.0001`
in channel units. The score is the absolute observed-minus-forecast residual
divided by that scale. At least **15 adjacent minutes above 1 in the same channel**
produce one advice item for that interval. Persistence restarts at gaps, room
changes and split boundaries. These fixed settings were selected before the
real replay; validation/test results do not refit them. Unlabelled training data
is a reference distribution, not confirmed normal data.

The existing deterministic checks and per-device IsolationForest still run.
`forecast-residuals.csv` records every target timestamp, forecast origin, horizon
step, channel, observation, forecast, signed residual, frozen scale and score.
Advice identifies the affected channel/room, sustained interval, score, model
version/hash and calibration hash, and suggests checking the history, sensor and
room conditions without inferring a cause. Changed model or calibration evidence
invalidates previous review IDs.

`report.json` adds the candidate hashes, contract, training residual hashes/scales,
window counts and TTM/IsolationForest advice-review comparison over the **same
validation dates and source dataset**. Their feature sets and eligible windows
differ: TTM uses complete temperature/humidity contexts; the forest uses hourly
RSSI/free-memory medians. A #26 export without `manual-labels.csv` supplies no
human labels; new advice remains unreviewed. Fill the generated review CSV and
rerun the same command with `--reviews <edited-csv>` into a new output directory.
Missing/incompatible inputs fail; incomplete validation review keeps acceptance
failed. The original eight historical cases remain a separate required replay
on the reviewed #19 dataset: the shorter #26 experiment does not contain all of
them, so its historical-reproduction gate also remains failed.

The initial unreviewed replay on October 20 used four devices, 250 training / 96 validation / 96
test windows and produced 84,864 channel residuals. TTM produced 12 training,
zero validation and 14 test advice items. That October 15-17 comparison had eight
IsolationForest and two deterministic validation items awaiting fresh review;
its false-positive rate was unknown and its acceptance remains **false** in the
retained original artifacts. The subsequent reviewed result is below. Zero TTM
validation alerts do not establish sensitivity or predictive accuracy. The
retained #19 replay still passes all eight historical findings and has identical
advice, reviews and scores, with 69 reviewed validation items and zero false
positives. Its report differs only in the recorded scikit-learn version
(`1.9.0` previously, `1.7.2` in the #26 laptop environment).

Reviewable artifacts are in `data/advisor-issue27-health-20261020/` (including
`review.md`, `report.json` and `verification.json`); the retained historical run
is `data/advisor-issue27-baseline-preserved-20261020/`. These files and the model
candidate are gitignored and must be retained separately. The independent
comparison/review-document command is `python data/issue27-verify.py`.
The final-code replay in `data/advisor-issue27-health-final-20261020/` has identical
report, advice, reviews, baseline scores and residual CSV. Local verification:
441 tests passed, Ruff lint/format and mypy passed; hosted CI has not run.
Runtime/API wiring remains #32; this change adds no runtime settings or dependencies.

An additional five-epoch fine-tuning experiment on October 20 retained the
original candidate and selected its checkpoint by validation MSE only. The best
was the first additional epoch (two epochs in total): validation temperature /
humidity MAE changed from 0.0756 degC / 0.5133 percentage points to 0.0737 /
0.4787, still behind last-value's 0.0686 / 0.4722. On the already inspected test
period, temperature MAE worsened from 0.1715 to 0.1757 degC and humidity improved
from 1.2687 to 1.2491 percentage points; last-value remains better at 0.1445 /
1.1053. Further epochs did not improve validation loss. Native/ONNX parity and
source-checkpoint preservation passed. This is not a fresh blind test or an
approved model, and no review verdicts or detector thresholds were changed.
Candidate/report: `artifacts/advisor-ttm-climate-continued5-20261020/`, including
`continuation-comparison.json`; comparison command:
`python data/issue27-finetune-compare.py`. Numerical forecast quality is measured
automatically; human review supplies event context and whether advice is useful.

On October 20 the user confirmed, **"The warnings are useful"**, in response to
the pending numbered review. This was recorded as `true-positive` usefulness
verdicts for all 36 displayed rows: 10 validation items and 26 new TTM
training/test items. The notes quote that statement and do not confirm suggested
causes. Review CSV and exact-ID/chat provenance:
`data/advisor-issue27-review-20261020/reviewed.csv` and `provenance.json`.

The fresh reviewed replay in `data/advisor-issue27-health-reviewed-20261020/`
has **10 reviewed validation items, zero unreviewed and zero false positives**
over two calendar days: **0.0/day**, below the required 1/day. The separate fresh
historical replay in `data/advisor-issue27-historical-reviewed-20261020/` passes
all eight named cases, with the original 69 reviewed validation items and zero
false positives over nine days. Its advice/review/score files remain identical.

The verified combined offline #27 acceptance result is **passed** in
`data/advisor-issue27-health-reviewed-20261020/acceptance-verification.json`.
`python data/issue27-verify-reviewed.py` asserts that only the 36 approved verdicts
and notes changed, both score CSVs and calibration/model/source hashes stayed
identical, and the historical cases and reviewed false-positive limits pass.
The short replay's original all-history CLI gate still returns false/exit 1
because its date range lacks the historical cases; the paired acceptance report
links both runs by SHA-256 instead of altering that gate. Models, thresholds,
original reports and dataset labels were not changed. All 28 focused
detector/review tests passed after recording the review.

This completes the offline #27 acceptance scope for these user-reviewed samples.
No TTM validation advice fired, so the review does not establish its sensitivity
to future events or independent root-cause accuracy. Test verdicts did not tune
the model. Deployment approval remains false and live TTM integration remains
#32. Keep these gitignored evidence artifacts separately from the code checkout.

### Advisor indoor climate replay

Replay the existing #18 dataset without querying InfluxDB:

```powershell
uv run edgeops-train advisor-climate `
  --input-dir data/advisor-issue18-reviewed-20261019 `
  --output-dir data/advisor-climate
```

The new directory contains `advice.json` and `report.json`, with source hashes,
thresholds, observed rooms and warm-up coverage. Existing outputs are never
overwritten. Each advice item includes its historical room, source device,
UTC evidence window, observation time and a concrete action.

- **Humidity/mould risk:** 120 consecutive minute means above 60% RH. Override
  the duration with `--humidity-minutes N`. Recovery at or below 55% rearms the
  rule; readings between 55% and 60% cannot trigger a new episode. Advice suggests
  10 minutes of ventilation if outdoor air is drier, or dehumidification and a
  damp/cold-surface check. The 60% threshold follows
  [EPA moisture guidance](https://www.epa.gov/mold/mold-course-chapter-2);
  duration and hysteresis are initial detector settings, not proof of mould.
- **Air quality:** 15 consecutive minute means at or below 59.8 on ingest's
  0–100 IAQ scale, where higher is better. Every minute must explicitly report
  `warmed_up=true`; false, missing or unknown flags break the window. Recovery at
  or above 69.8 rearms the rule. Evidence includes IAQ change and, when present
  throughout the window, gas-resistance start/end/change in ohms. Gas alone does
  not identify a pollutant or trigger advice. Suggested actions are a source
  check and extraction or ventilation when outdoor air quality permits. The
  scale is documented in the
  [ingest contract](https://github.com/hengsti/ingestion-service/blob/main/docs/messages-and-routing.md#iaq-scale).

Each source device needs its own continuous evidence; readings from multiple
devices are not stitched together to satisfy a duration. Missing minutes,
room changes and explicitly invalid clock flags break windows. Windows also
stop at train/validation/test boundaries. Episode state and a shared room
cooldown permit at most one advice item per room in any rolling hour, across
both rules and all devices. A persistent episode is not repeated until recovery;
gaps do not establish recovery. State lasts for the replay invocation.

Without a forecast candidate this implements the deterministic part of #20.
Estimated wall temperatures are absent. Outdoor moisture suitability remains
unknown unless matching weather observations are supplied as described below. The optional
offline TTM consumer below implements #28; live forecast integration belongs to
#32. Historical replay
checks the data and detector behavior; advice usefulness still needs human review.
The separate `advisor-health` acceptance gate requires a reviewed true positive
for each intended case, all eight historical findings, a fitted baseline for
each device and fewer than one false advice item per validation day. That command
returns **1** when acceptance is pending or failed, while retaining its reports;
it returns **0** only when acceptance passes.

#### Outdoor weather for ventilation (#29)

The optional integration uses only OpenWeatherMap's
[Current Weather API](https://openweathermap.org/api/current). Configure the home
postcode and API key in `.env`; both are required to enable live weather:

```dotenv
ADVISOR_WEATHER_POSTCODE=1030,AT
ADVISOR_WEATHER_API_KEY=your-openweathermap-api-key
```

The postcode includes the two-letter country code (`1030,AT` for the requested
Austrian location). Obtain a key with Current Weather access from OpenWeatherMap
and restart the service after editing `.env`. Compose already loads this file.
The key is kept out of reports and request logs. No key or no postcode leaves
deterministic indoor advice enabled with unknown outdoor suitability.

The live state report shares one observation across all rooms, fetching at most
once every 10 minutes per service process, including failed attempts. Requests
have a three-second timeout. Missing/malformed responses, API errors and weather
older than 30 minutes are unavailable; a failed refresh discards the old cache.
Weather must be at or before the advice observation time and at most 30 minutes
old at that time. The live report replays a day of history, so its older findings
can remain conditional even when current weather is available.

Ventilation advice compares absolute humidity in g/m³ using indoor and outdoor
temperature and relative humidity. Lower outdoor absolute humidity supports
ventilation for moisture reduction; equal or higher values suggest dehumidification.
Missing indoor temperature also leaves the comparison unknown. Evidence records
the provider, location, UTC weather timestamp, temperatures, relative humidities,
absolute humidities and their indoor-minus-outdoor difference. Humidity/IAQ
thresholds, recovery and the shared one-hour room cooldown are unchanged.
**Weather does not establish outdoor pollutant safety:** ventilation remains
conditional on outdoor air quality, including when outdoor air is drier.

Offline replay never calls the weather API or loads the weather key. To use
recorded observations, supply a JSON file with the following shape:

```json
{
  "observations": [
    {
      "provider": "openweathermap",
      "location": "1030,AT",
      "observed_at": "2026-10-19T10:00:00+00:00",
      "temperature_c": 8.0,
      "humidity_percent": 85.0
    }
  ]
}
```

```powershell
uv run edgeops-train advisor-climate `
  --input-dir data/advisor-issue18-reviewed-20261019 `
  --weather-observations data/outdoor-weather.json `
  --output-dir data/advisor-climate-weather
```

Each finding uses the most recent observation at or before its own timestamp,
subject to the same freshness check. Present-day observations cannot enrich past
findings. The report records the input weather file's SHA-256 alongside dataset
hashes. This option also works with `--forecast-candidate` and adapts its existing
humidity ventilation action without changing model predictions.

#### Proactive climate forecasts (#28)

Reuse the validated #26 climate candidate and its exact matching chronological
dataset, without training or changing either source:

```powershell
uv run --no-sync edgeops-train advisor-climate `
  --input-dir data/advisor-issue26-climate-20261010-20261019 `
  --forecast-candidate artifacts/advisor-ttm-climate-20261020 `
  --output-dir data/advisor-issue28-climate-20261021
```

Choose a new output directory for each run. The command retains `advice.json`
and `report.json`, and adds `forecasts.json`: per-device forecasts within one
stable room, with temperature in degC, relative humidity in percent, model
version/SHA-256, context start/stop, last input, forecast origin and UTC target
timestamps. Multiple devices are not averaged or stitched together.

The candidate needs its ONNX file, digest, embedded/JSON manifest and passing
#26 parity report. The source dataset hashes must match its manifest. The
consumer reuses #26's 512-context/96-target windows, including their 96-minute
origin stride, rather than introducing a separate windowing implementation.
Missing channels, gaps, missing/false warm-up flags, explicitly invalid sensor
clocks, resets, room changes and chronological split boundaries break windows.
Incomplete windows produce no forecast or advice. Invalid candidate/data checks
fail without publishing a report; nonphysical predicted humidity also fails.

Aggregate timestamps denote minute **starts**. The forecast origin is the end
of the completed input context, one minute after the last input timestamp.
The first target starts at that origin (lead zero); the consumer retains leads
0 through 60, so index 30/60 is exactly 30/60 minutes after the origin. The
evidence window contains observed context only, never the future crossing.

An upward humidity crossing above the existing 60% threshold, with current RH
at or below 60%, can produce advice at issuance for leads 1–60 minutes. An
already high reading or lead-zero crossing produces no proactive advice.
Evidence includes the threshold, current/predicted RH, estimated minutes and
target timestamp. Forecast advice shares the deterministic rules' rolling
60-minute room cooldown across all devices; observed RH at or below 55% rearms
its episode. It suggests a humidity/ventilation check, without inferring
cooking/heating causes or assuming outdoor conditions are suitable.

The report compares TTM, last-value and a daily seasonal baseline on the same
validation/test windows. The seasonal baseline averages unique training
observations per device/room and UTC minute of day; overlapping contexts do not
overweight observations. Its sample count and profile hash identify the fitted
baseline. Missing seasonal coverage excludes a window from **all three**
comparisons and is counted explicitly, without filling missing values. Existing
60%/55% thresholds are fixed rules, not fitted on held-out data. TTM preprocessing
remains inside the validated #26 graph. Metrics include MAE/RMSE by lead/channel,
mean MAE over leads 30–60, and raw humidity-crossing true/false positives,
misses and timing MAE on matched crossings. Crossing metrics describe numeric
forecasts before advice cooldown/hysteresis; already high contexts are excluded.

The local October 10–19 replay on 2026-10-21 produced **442 forecasts** in
bedroom, buero, kitchen and livingroom: 250 training, 96 validation and 96 test
windows. Both held-out comparisons covered **96/96** windows. The seasonal
profile used 28,608 unique training samples. After shared cooldown/hysteresis,
the replay emitted 5 forecast and 8 observed-humidity advice items.

| Split | Model | Temperature MAE, 30–60 min (degC) | RH MAE, 30–60 min (pp) | Crossing TP / FP / misses | Matched timing MAE (min) |
| --- | --- | --- | --- | --- | --- |
| Validation | TTM | 0.079 | 0.468 | 2 / 1 / 1 | 24.0 |
| Validation | Last-value | 0.073 | 0.432 | 0 / 0 / 3 | no matches |
| Validation | Daily seasonal | 0.760 | 3.026 | 0 / 4 / 3 | no matches |
| Test | TTM | 0.162 | 1.273 | 3 / 3 / 6 | 17.7 |
| Test | Last-value | 0.132 | 1.088 | 0 / 0 / 9 | no matches |
| Test | Daily seasonal | 0.855 | 5.371 | 2 / 0 / 7 | 24.0 |

Crossing evaluation has 51 eligible validation and 31 eligible test windows;
the remaining origins already exceed 60%. Raw reports are in the new output
directory above. The candidate's ONNX SHA-256 is
`7ae6ad6b85bc9bd047fc23df804d31704a66dcfcb000c1188039506a4953cef4`;
the fitted seasonal profile SHA-256 is
`93ef01e9165db3972efb68291dada59ac40781bc999231152609edad1506bd49`.
TTM loses to last-value on both channels/splits, and crossing errors remain.
`deployment_approved=false` is retained; completing the offline consumer does
not approve the model or establish advice usefulness. Runtime/API delivery,
fresh Pi measurements and outdoor integration are outside this replay.

Focused verification:

```powershell
uv run --no-sync pytest tests/test_advisor_climate.py tests/test_advisor_climate_replay.py `
  tests/test_advisor_climate_forecast.py tests/test_advisor_forecast.py `
  tests/test_advisor_residuals.py tests/test_state_report.py -q
```

The final local suite passed **470 tests**; Ruff lint/format and mypy passed.
The suite reported 67 dependency warnings, including deprecations and the
existing incident classifier's scikit-learn artifact-version warning. Independent
code review found no issues. All 442 real forecast alignments, split boundaries,
room cooldowns and unchanged dataset/candidate hashes were also checked. Hosted
CI, live API delivery and fresh Pi validation were not run for this offline change.

### Advisor maintenance estimates

Replay #18's root-disk and device-memory history locally:

```powershell
uv run edgeops-train advisor-maintenance `
  --input-dir data/advisor-issue18-reviewed-20261019 `
  --output-dir data/advisor-maintenance
```

The new directory contains `state-report.json` with each series' current value,
threshold, daily slope, ETA in days, uncertainty range and status; `advice.json`
contains only actions due within the horizon. `report.json` records input hashes,
thresholds and battery backtest cases. Existing reports and source datasets stay
unchanged, source changes during replay are rejected, and publication is atomic.

The development-only baseline uses the installed
[SciPy Theil–Sen estimator](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.theilslopes.html)
on up to 30 consecutive completed UTC daily medians, requiring at least seven
days. Root-disk days need 144 distinct five-minute aggregates, and device-memory
days need 720 distinct one-minute aggregates with uptime evidence. Disk identity
retains its host and original mount/device tags. Each source series and dataset
split stays separate; gaps are not filled. Days crossing split boundaries are
excluded. An observed uptime reset/discontinuity excludes that device's day,
breaking its trend. A battery rise of at least 10 percentage points or disk drop
of at least five versus the preceding three-day median also restarts history.
These intervention thresholds are initial heuristics.

Defaults are operator thresholds, not learned failure limits:

| Metric | Default threshold | Override |
| --- | --- | --- |
| Hue battery level | 10% remaining | `--battery-threshold` |
| Root disk usage | 90% used | `--disk-threshold` |
| Device free memory | 65,536 bytes | `--memory-threshold` |

The ETA starts from the last daily median and assumes a constant future trend.
Its range propagates the 95% confidence limits of the fitted **slope**; it is
not a calibrated prediction interval for the actual crossing time. An interval
extending to no crossing has a null upper endpoint and suppresses advice. A
recovering/flat trend has no ETA, and missing daily history or data at least a
day old is explicitly insufficient/stale. An already reached threshold has ETA
zero. Otherwise advice requires the entire finite ETA range to be within
`--horizon-days` (default 30). Long ETAs remain visible for inspection and should
not be treated as reliable long-term plans.

The #18 export currently excludes Hue measurements. Supply optional battery
history with `--battery-csv <file>`, using this format:

```csv
time,entity_id,battery_percent
2026-09-10T00:00:00Z,bridge-id/device-id,25
2026-09-11T00:00:00Z,bridge-id/device-id,25
```

Use a stable bridge/device identity, explicit UTC timestamps within the dataset
range, and actual observed percentages from `huebridge_device_power.battery_level`.
Export a measurement on each observed day; do not forward-fill gaps. Repeated
raw observations within a day are reduced to a median. At least seven consecutive
observed days are needed, so the two illustrative CSV lines above cannot produce
a trend. The battery source is hashed with the Parquet inputs.

Backtesting uses only observations preceding each forecast origin and evaluates
the first subsequent observed daily threshold crossing within the same split
and battery episode. A replacement or missing day censors that forecast. Cases
from validation/test are reported with predicted/actual durations and relative
error; the ±20% criterion passes only if every evaluated case meets it. With no
observed crossing it remains unknown, including when no battery input is supplied.
Daily observations limit timing resolution; test fixtures do not establish real
battery accuracy. Exit zero means replay succeeded, not that battery acceptance
passed.

This implements #21's robust baseline and local state report. CPU activity and
container-health events are assessed separately from battery/disk/memory depletion
trends. No new dependencies or automatic maintenance actions are introduced.

### Advisor host workload replay

Issue #30 adds optional offline assessment of observed workload conditions:

```powershell
uv run edgeops-train advisor-maintenance `
  --input-dir data/advisor-issue18-reviewed-20261019 `
  --forecast-candidate artifacts/advisor-ttm-host-cpu-20261020 `
  --output-dir data/advisor-workload
```

The validated CPU artifact retains its 512-input / 96-output contract. This
consumer uses only its first five-minute expectation, then compares it with the
arriving completed `cpu-total` / `usage_active_mean` aggregate. Every expectation
uses 42h40m of prior observations. Identities, chronological splits, gaps, missing
or invalid CPU and supplied host reset/deploy events break context; nothing is
imputed. Unsupported artifacts, changed source hashes or failed integrity/parity
checks reject publication. Incomplete history produces explicit unavailable
evidence. A fresh context must accumulate after each split or interruption.

The Advisor distinguishes `expected-workload`, `unusual-activity`,
`resource-pressure` and `insufficient-evidence`. A CPU aggregate at or above
90% (`--cpu-threshold`) or an absolute residual above the host's frozen training
99th percentile is unusual evidence. Scores divide that residual by the training
scale, with the existing numeric tolerance as a floor. Calibration uses training
windows only, separately for TTM and last-value; held-out data never fits it.
Scores are not probabilities, and confidence remains null.

Advice requires three consecutive completed five-minute samples
(`--workload-minutes 15`). High CPU or a deviation alone produces informational
advice. A warning additionally needs observed unhealthy/OOM service evidence at
each sample. Memory usage is supporting context; no unapproved memory-pressure
threshold is inferred. Missing health is explicitly unavailable. Observed container
identity, version, restart or uptime changes break health persistence without
erasing unrelated host CPU history. Recovery ends an episode. Host reset/deploy
markers are absent from the retained dataset, so hidden boundaries remain unknown.
An activity is called a build only when supplied activity evidence identifies it;
otherwise its cause stays unknown. Expected workload does not certify service health.

Existing maintenance outputs remain present. `state-report.json` gains latest
resource states; a last sample older than the dataset end is marked stale.
`advice.json` gains TTM workload findings, with observed/expected values aligned
to the same target, first detection time, duration, health context and model and
calibration hashes. `resource-observations.csv` exposes every observation and
availability status. `resource-advice-review.csv` compares TTM, CPU rules alone
and last-value on identical periods; `report.json.resources` records calibration,
coverage, comparisons, sources and acceptance. Publication uses a new directory
and preserves the original datasets, models and reports.

Optional reviewed context uses `--workload-events events.csv` and the existing
manual-label format:

```csv
entity_type,entity_id,start,stop,label,note
host,my-pi,2026-10-02T10:00:00Z,2026-10-02T10:30:00Z,build,Operator confirmed application build
```

Use aligned UTC timestamps and a host identity present in the dataset. Supported
labels are `build`, `normal-activity`, `deploy`, `reset` and `resource-pressure`.
The last label is evaluation truth only and never informs detection. Deploy/reset
intervals exclude CPU context through their end boundary. Manual context stays
separate from weak labels and does not prove a cause.

Review a copy of `resource-advice-review.csv`, setting `verdict` to
`true-positive` or `false-positive` and adding a reason in `note`. Rerun into a new
directory with the same inputs plus `--resource-reviews <reviewed.csv>`. Review
IDs bind source, model, calibration, policy and finding evidence; changed evidence
cannot inherit old verdicts. Acceptance needs reviewed normal and incident cases,
all validation findings reviewed for all three methods, fewer than one false
positive warning per day and known incidents detected within 15 minutes. With no
known incident, detection delay and acceptance remain unverified. Synthetic tests
check implementation behavior; they do not establish historical usefulness.

The first retained October 1–9 replay produced 58 TTM informational findings and
zero warnings; rules and last-value produced 3 and 7 informational findings,
respectively. All need review. The operator reported no known incident in that
period, so these counts do not establish warning quality or incident detection.
The candidate and `deployment_approved=false` are unchanged. Its exact existing
Pi evidence is reused (p95 24.49 ms; peak RSS 77.72 MB), with no new Pi run. Live
host/history inputs belong to #31 and runtime model wiring to #32.

### Advisor native training and ONNX export

Issue #26 uses IBM Granite TTM-r2, the first candidate recommended in
[#17](https://github.com/hengsti/edgeops-ai-service/issues/17). This workflow
produces candidates; model selection and forecast-quality approval remain separate.
Pi latency/process-memory measurements for the current candidates are recorded below.
The incident classifier keeps its separate 46-feature contract and exporter.

On the Windows laptop, use Python 3.13 and install the optional training group:

```powershell
uv sync --locked --group advisor
uv run --no-sync hf download ibm-granite/granite-timeseries-ttm-r2 `
  config.json model.safetensors `
  --revision d6a79570cac0f33d526601cd3a0fc7c80a8f9a2f `
  --local-dir artifacts/ttm-r2-source

uv run --no-sync edgeops-train advisor-forecast `
  --input-dir data/advisor-issue18-reviewed-20261019 `
  --checkpoint artifacts/ttm-r2-source `
  --sha256 a706726a7eb01bbcb42994b7dcb3c06ea9557898dbae8d480eb04fe8ccb89710 `
  --profile host-cpu --model-version advisor-ttm-host-cpu-20261020 `
  --epochs 1 --output-dir artifacts/advisor-ttm-host-cpu-20261020
```

The immutable revision above is the 512-context/96-target TTM-r2 checkpoint.
Only trusted local `save_pretrained` directories with `config.json` and
`model.safetensors` are loaded, with the recorded weights SHA-256 checked first.
No joblib intermediate is used. The command does not fetch model code or weights.
PyTorch, Granite TSFM, ONNX and ONNX Script stay in the laptop environment;
`uv sync --locked --no-dev --extra ml` and the production image exclude the training group.
Granite TSFM 0.3.5 caps scikit-learn below 1.8, so the laptop environment can warn
when reading the older incident joblib artifact trained with 1.9. Keep the
approved incident ONNX artifact for inference; do not resave the source joblib.

The new output directory contains `model.onnx`, `model.onnx.sha256`,
`manifest.json`, `validation-report.json` and the selected native `checkpoint/`
(`config.json`, `model.safetensors`, `forecast-contract.json`). Existing outputs
and source directories cannot be overwritten. Native fine-tuning uses AdamW
(learning rate 0.0001, batch size 1), seed 7 and one CPU thread. `--epochs`
defaults to one; the lowest validation-loss epoch is retained. Test windows
are evaluated only after epoch selection. The retained checkpoint supports
later fine-tuning through the same command with its weights hash and profile;
each run starts a new optimizer. Source-file and dataset hashes, library versions,
training time and chronological forecast origins are recorded.

For conversion without training, use the same command with `--export-only` and
a new output directory. It validates against the same historical inputs and
checks every source checkpoint file remains byte-for-byte unchanged. It writes
no checkpoint copy. Hash the retained `checkpoint/model.safetensors` when
exporting a fine-tuned candidate instead of using the pretrained hash above.

The embedded `edgeops_advisor_manifest` and JSON manifest define
`advisor-forecast-1`. Each artifact supports exactly one of these two profiles:

| Profile | Identity and channel order | Cadence | Context / horizon with the checkpoint above |
| --- | --- | --- | --- |
| `climate` | One device in one unchanged room: `temp_c_mean` (degC), `rel_hum_perc_mean` (%) | 1 minute | 512 minutes / 96 minutes, including the 30- and 60-minute targets |
| `host-cpu` | One host, `cpu-total`: `usage_active_mean` (%) | 5 minutes | 512 samples (42h40m) / 96 samples (8 hours) |

Inputs are `past_values`, float32 `[1, context_length, channels]`; outputs are
`forecast`, float32 `[1, prediction_length, channels]` in the same physical units.
The first target follows the last input by one cadence. TTM performs its own
per-channel standard scaling from that input context inside both native and ONNX
graphs. There is no globally fitted scaler or future-data normalization.
Only regular, complete UTC aggregates are accepted. Windows never cross
identities, room changes, gaps, device uptime resets or chronological splits.
Climate windows also require an explicit true `warmed_up_last` throughout.
Missing channels/metadata produce unavailable windows; nothing is filled in.
Windows advance by one prediction horizon within each uninterrupted segment.

`edgeops_ai.forecasting.ForecastModel(path, expected_sha256)` is the shared
inference entry point for #22/#32 and consumers #27/#28/#30. It validates the
artifact hash, embedded manifest and exact graph tensor names/types/shapes.
`predict(values, utc_timestamps)` uses the same `prepare_input` function as
training. Callers supply the manifest channel order and enforce the identity,
room, reset, completed-window and data-quality checks above. The climate
artifact forecasts devices; #28 remains responsible for per-room delivery.
The resource artifact supports host CPU point forecasts over eight hours;
it does not support battery/disk/device-memory ETAs or extrapolation beyond
its horizon. These outputs carry no calibrated confidence or incident category.

Export uses PyTorch's current exporter with ONNX opset 18, then full ONNX graph
checking and ONNX Runtime CPU execution. Validation and test forecasts must
match native outputs elementwise within `abs(error) <= 0.0001 + 0.0001 * abs(native)`.
The report includes maximum output drift, MAE/RMSE for every horizon/channel
and a last-value baseline on the identical held-out windows. Conversion,
contract, data-integrity or parity failures exit nonzero and publish no
candidate directory. Exit zero validates conversion, not predictive quality or
Pi deployment readiness; `deployment_approved` remains false.

An offline smoke test creates a tiny chronological Parquet dataset and native
TTM fixture, then runs training/export/validation, export-only and later native
fine-tuning without downloading a checkpoint:

```powershell
uv run --group advisor pytest tests/test_advisor_forecast.py -q
```

CI runs this check on Windows. Production ONNX inference requires the `ml` extra,
including NumPy via ONNX Runtime; it does not import
PyTorch, Granite TSFM, PyArrow or ONNX conversion libraries.

The reviewed September 10–October 19 dataset lacks `warmed_up_last` in its original
training period. A separate climate evaluation view uses existing October 10–19
observations, training before October 15, validation before October 17 and test
afterward. `data/advisor-issue26-climate-20261010-20261019/` records the parent hashes
and split provenance. Only the selected time range and split column changed;
measurements, labels and the original reviewed export were verified unchanged.
This is a new chronological forecast experiment using the original export's
test period; it does not replace that export or its earlier evaluation results.
The reproducible local derivation is `data/issue26-climate-resplit.py`.
No candidate is promoted by this CLI.

Local Windows validation on 2026-10-20 produced
`artifacts/advisor-ttm-host-cpu-20261020/`: 42 training, 21 validation and 12 test
windows, one native training epoch (0.97 seconds), and maximum native/ONNX drift
0.0000344. Test MAE averaged across the 96 horizons was **2.46 percentage points**,
versus **1.63** for last-value; validation MAE was 1.68 versus 1.53. This candidate
does not improve that baseline. Its ONNX SHA-256 is
`d0caefe7d0d7d70709f74421ea3413fd4614fa9e1d21242a2a808a0ca802a107`.
The retained checkpoint was also exported without training to
`artifacts/advisor-ttm-host-cpu-export-20261020/`, with source hashes unchanged.
Both directories contain their own validation reports. Generated candidates
remain local and ignored by git; they are not approved release artifacts.

The real climate run at `artifacts/advisor-ttm-climate-20261020/` used 250 training,
96 validation and 96 test windows, one CPU epoch (3.88 seconds), and maximum
native/ONNX drift 0.000153 within the documented elementwise tolerances. Test MAE
averaged across horizons was 0.171 degC / 1.269 humidity percentage points,
versus last-value 0.144 / 1.105. This candidate also remains unapproved. Its ONNX
SHA-256 is `7ae6ad6b85bc9bd047fc23df804d31704a66dcfcb000c1188039506a4953cef4`.

Inference was verified on Windows x86_64 in a clean environment containing only
the locked ML runtime dependencies. Reproduce that dependency check without
installing any training packages into it:

```powershell
uv export --locked --no-dev --extra ml --no-emit-project --quiet `
  --output-file data/advisor-runtime-requirements.txt
uv venv data/advisor-runtime
uv pip sync --python data/advisor-runtime/Scripts/python.exe `
  data/advisor-runtime-requirements.txt
$env:PYTHONPATH = 'src'
@'
import importlib.util, json
from datetime import UTC, datetime, timedelta
from pathlib import Path
import numpy as np
from edgeops_ai.forecasting import ForecastModel
assert all(importlib.util.find_spec(p) is None for p in
           ("torch", "tsfm_public", "sklearn", "pyarrow", "onnx", "onnxscript"))
root = Path("artifacts/advisor-ttm-host-cpu-20261020")
manifest = json.loads((root / "manifest.json").read_text())
model = ForecastModel(root / "model.onnx", manifest["sha256"])
times = [datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=5*i) for i in range(512)]
assert model.predict(np.full((512, 1), 30), times).shape == (1, 96, 1)
print("Runtime-only inference passed")
'@ | .\data\advisor-runtime\Scripts\python.exe -
```

This proves dependency separation on the laptop; hosted CI remains unverified.

Pi validation on 2026-10-20 used the actual Raspberry Pi 4 Model B Rev 1.4,
ARM64, Python 3.13.16 and ONNX Runtime 1.31.0 from the running service's image
`sha256:0931edb24210d5724437adf030ee5bf54a25d15f2b158af13fc27bea2580d82d`.
Each candidate ran in a temporary, read-only, network-disabled container with
one CPU and one inference thread. Native references came from the first,
middle and last real held-out windows. No training packages were present.
After five warm-up calls, 50 calls measured the complete `ForecastModel.predict`
path, including shared input validation. Linux `ru_maxrss` measured peak total
process memory, including Python/NumPy/ONNX Runtime, rather than weights size.

| Candidate | Median / p95 / maximum latency | Peak process memory (decimal MB) |
| --- | --- | --- |
| Host CPU | 12.51 / 24.49 / 44.14 ms | 77.72 |
| Climate | 10.95 / 11.99 / 12.74 ms | 77.54 |

Both met the measured p95 <50 ms / peak RSS <100 MB targets; Pi output drift
from native references was at most 0.0000344. Raw reports and input/model hashes
are at `data/issue26-pi-validation-20261020/`. The exact candidates also contain
`pi-benchmark.json`, linked by hash from `validation-report.json`.

The 27-sample host/health observation recorded unchanged production container
states/restart counts and HTTP 200 for AI and collector health throughout.
Median host CPU usage was 53.5% before, 88.7% during the burst and 44.2% afterward.
AI health p95 rose from 16.9 to 145.3 ms during the burst, then returned to 21.8 ms;
collector health p95 was 7.8 / 31.0 / 14.8 ms. This short observation includes
background workload and container startup; it does not establish causal,
long-term or MQTT-delivery impact. Docker memory cgroups are unavailable on
this host, so Docker's zero-byte memory display was not used. No host settings,
production files, running services or deployment were changed.

Re-run the inference-only command inside an environment with the ML runtime
dependencies and the exported candidate/reference inputs:

```sh
python -m edgeops_ai.training.advisor_benchmark \
  --model host-cpu/model.onnx \
  --sha256 d0caefe7d0d7d70709f74421ea3413fd4614fa9e1d21242a2a808a0ca802a107 \
  --inputs host-cpu/samples.npz --iterations 50
```

`data/issue26-pi-bundle/` contains the tested source, both candidates, NPZ inputs
and a provenance manifest; `data/issue26-prepare-pi.py` reproduces it on the laptop
using native checkpoints and actual held-out windows. The NPZ contains float32
`values`, UTC `times` and native `expected` outputs; pickle loading is disabled.
Benchmark exit zero means measurement/parity succeeded. Check
`resource_targets_met` separately; non-Linux runs cannot establish peak RSS.
The CLI does not approve deployment. Both current candidates still lose to
last-value forecasts, so further model selection stays separate from #26's
completed training/export integration. No forecast consumer was deployed.

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
