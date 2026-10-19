"""Export the monitoring aggregate contract as labelled, chronological Parquet tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from contextlib import ExitStack
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import httpx
import pyarrow as pa
import pyarrow.parquet as pq

from edgeops_ai.training.reports import write_report

KINDS = ("device", "host", "container", "room")
EXCLUDED_DEVICES = {"stresstest", "livingroom-01"}
CLIMATE_FIELDS = {
    "temp_c",
    "rel_hum_perc",
    "pressure_hpa",
    "gas_ohm",
    "iaq_score",
    "dew_point_c",
    "heat_index_c",
}
FAILURE_COUNTERS = {
    "sensor_read_failures_total_last": "sensor-read-failure",
    "samples_failed_total_last": "sample-failure",
    "dropped_total_last": "samples-dropped",
}
EVENT_COLUMNS = ("entity_type", "entity_id", "start", "stop", "label", "source", "note")
MANUAL_COLUMNS = tuple(c for c in EVENT_COLUMNS if c != "source")
SCHEMA = pa.schema(
    [
        ("time", pa.timestamp("us", tz="+00:00")),
        ("entity_id", pa.string()),
        ("measurement", pa.string()),
        ("field", pa.string()),
        ("value", pa.float64()),
        ("text", pa.string()),
        ("dimensions", pa.string()),
        ("interval_minutes", pa.int16()),
        ("split", pa.string()),
        ("weak_labels", pa.list_(pa.string())),
        ("manual_label", pa.string()),
    ]
)
# Dates in #18 are review hints; their timezone and exact short-gap bounds are unspecified.
KNOWN_EVENT_DATES = (
    "2026-06-20",
    "2026-07-04",
    "2026-09-11",
    "2026-09-12",
    "2026-09-13",
    "2026-09-14",
    "2026-09-30",
    "2026-10-02",
    "2026-10-07",
)


def timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.utcoffset() != timedelta(0):
        raise ValueError("Use explicit UTC timestamps (Z or +00:00)")
    return parsed.astimezone(UTC)


def boundary(value: str) -> datetime:
    try:
        parsed = timestamp(value)
        if parsed.minute % 5 or parsed.second or parsed.microsecond:
            raise ValueError("Boundaries must align to five minutes")
        return parsed
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def read_records(
    client: httpx.Client, bucket: str, start: datetime, stop: datetime
) -> Iterator[dict[str, str]]:
    query = (
        f"from(bucket: {json.dumps(bucket)}) "
        f"|> range(start: {start.isoformat()}, stop: {stop.isoformat()}) "
    )
    if bucket == "sensors_1m":
        query += (
            '|> keep(columns: ["_time", "_measurement", "_field", "_value", "device_id", "room"])'
        )
    with client.stream(
        "POST", "/api/v2/query", json={"query": query, "dialect": {"annotations": ["datatype"]}}
    ) as response:
        response.raise_for_status()
        header = None
        datatypes = None
        for row in csv.reader(response.iter_lines()):
            if row and row[0] == "#datatype":
                datatypes = row
                continue
            if not row or row[0].startswith("#"):
                continue
            if "error" in row and "reference" in row:
                raise ValueError("InfluxDB query returned a stream error")
            if "_time" in row and "_value" in row:
                header = row
            elif header:
                if len(row) != len(header):
                    raise ValueError("InfluxDB returned a malformed CSV row")
                if datatypes is None or len(datatypes) != len(header):
                    raise ValueError("InfluxDB response is missing datatype annotations")
                yield {
                    **dict(zip(header, row, strict=True)),
                    "_value_type": datatypes[header.index("_value")],
                }
            else:
                raise ValueError("InfluxDB returned an unexpected CSV response")


def normalize(
    records: Iterator[dict[str, str]], bucket: str, start: datetime, stop: datetime
) -> dict[str, list[dict[str, Any]]]:
    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in records:
        measurement, field = record["_measurement"], record["_field"]
        if bucket == "sensors_1m":
            if measurement not in {"bme680", "device_status"}:
                continue
            entity_id, kind, cadence = record.get("device_id", ""), "device", 1
            if entity_id in EXCLUDED_DEVICES or field.startswith(("ip_address_", "ssid_")):
                continue
            dimensions = {"room": record["room"]} if record.get("room") else {}
        else:
            if measurement in {"cpu", "mem", "disk"}:
                entity_id, kind = record.get("host", ""), "host"
            elif measurement in {
                "docker_container_cpu",
                "docker_container_status",
                "docker_container_health",
            }:
                host, name = record.get("host", ""), record.get("container_name", "")
                if not host or not name:
                    raise ValueError("Container aggregates need host and container_name")
                entity_id, kind = f"{host}/{name}", "container"
            else:
                continue
            cadence = 5
            dimensions = {
                k: v
                for k, v in record.items()
                if v
                and not k.startswith("_")
                and k not in {"", "result", "table", "host", "container_name"}
            }
        if not entity_id or not field:
            raise ValueError("Aggregate rows need a stable entity ID and field")
        time = timestamp(record["_time"])
        if not start <= time < stop or time.second or time.microsecond or time.minute % cadence:
            raise ValueError("Aggregate timestamp is outside its query or cadence")
        raw = record["_value"]
        value, text = None, None
        if record["_value_type"] in {"double", "long", "unsignedLong"}:
            value = float(raw)
            if not math.isfinite(value):
                raise ValueError("Aggregate values must be finite")
        else:
            text = raw
        row = {
            "time": time,
            "entity_id": entity_id,
            "measurement": measurement,
            "field": field,
            "value": value,
            "text": text,
            "dimensions": json.dumps(dimensions, sort_keys=True),
            "interval_minutes": cadence,
        }
        key = kind, entity_id, time, measurement, field, row["dimensions"]
        if key in unique and unique[key] != row:
            raise ValueError(f"Conflicting aggregate values after tag normalization: {key}")
        unique[key] = row
    tables: dict[str, list[dict[str, Any]]] = {kind: [] for kind in KINDS}
    for key, row in sorted(unique.items()):
        tables[key[0]].append(row)
    tables["room"] = room_records(tables["device"])
    return tables


def room_records(devices: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], dict[str, dict[str, dict[str, Any]]]] = defaultdict(dict)
    for row in devices:
        room = json.loads(row["dimensions"]).get("room")
        field, _, statistic = row["field"].rpartition("_")
        if (
            row["measurement"] == "bme680"
            and room
            and field in CLIMATE_FIELDS
            and statistic in {"mean", "min", "max", "count"}
            and row["value"] is not None
        ):
            groups[(row["time"], room, field)].setdefault(row["entity_id"], {})[statistic] = row
    result = []
    for (_, room, field), members in sorted(groups.items()):
        for statistic in ("count", "min", "max", "mean"):
            rows = [stats[statistic] for stats in members.values() if statistic in stats]
            if not rows:
                continue
            if statistic == "mean":
                weighted = [
                    stats
                    for stats in members.values()
                    if "mean" in stats and "count" in stats and stats["count"]["value"] > 0
                ]
                if len(weighted) != len(rows):
                    continue
                count = sum(stats["count"]["value"] for stats in weighted)
                value = (
                    sum(stats["mean"]["value"] * stats["count"]["value"] for stats in weighted)
                    / count
                )
            else:
                values = [row["value"] for row in rows]
                value = (
                    sum(values)
                    if statistic == "count"
                    else (min(values) if statistic == "min" else max(values))
                )
            result.append(
                {
                    **rows[0],
                    "entity_id": room,
                    "field": f"{field}_{statistic}",
                    "value": value,
                    "dimensions": "{}",
                }
            )
    return result


def load_manual_labels(path: Path | None, start: datetime, stop: datetime) -> list[dict[str, Any]]:
    if path is None:
        return []
    labels: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not set(MANUAL_COLUMNS).issubset(reader.fieldnames or []):
            raise ValueError(f"Manual labels require columns: {', '.join(MANUAL_COLUMNS)}")
        for row in reader:
            if any(row.get(column) is None for column in MANUAL_COLUMNS):
                raise ValueError("Manual label CSV has an incomplete row")
            if not (row.get("label") or "").strip():
                continue
            begin, end = timestamp(row["start"]), timestamp(row["stop"])
            cadence = 5 if row["entity_type"] in {"host", "container"} else 1
            if (
                row["entity_type"] not in KINDS
                or not row["entity_id"]
                or not row["note"].strip()
                or not start <= begin < end <= stop
            ):
                raise ValueError("Manual labels need an entity, review note and in-range interval")
            if any(t.second or t.microsecond or t.minute % cadence for t in (begin, end)):
                raise ValueError("Manual label boundaries must align to the entity cadence")
            label = {
                **row,
                "start": begin,
                "stop": end,
                "label": row["label"].strip(),
                "source": "manual-review",
            }
            for other in labels:
                if (
                    other["entity_type"] == label["entity_type"]
                    and other["entity_id"] == label["entity_id"]
                    and max(other["start"], begin) < min(other["stop"], end)
                ):
                    raise ValueError("Manual label intervals must not overlap for the same entity")
            labels.append(label)
    return labels


def weak_events(
    tables: dict[str, list[dict[str, Any]]],
    previous: dict[tuple[str, str], tuple[datetime, float]],
    last_seen: dict[str, datetime],
) -> list[dict[str, Any]]:
    events = []

    def event(
        kind: str, entity: str, start: datetime, stop: datetime, label: str, note: str = ""
    ) -> None:
        events.append(
            {
                "entity_type": kind,
                "entity_id": entity,
                "start": start,
                "stop": stop,
                "label": label,
                "source": "aggregate-rule",
                "note": note,
            }
        )

    sensor_times: dict[str, set[datetime]] = defaultdict(set)
    resets = {
        (row["entity_id"], row["time"], row["field"].removesuffix("_resets"))
        for row in tables["device"]
        if row["field"].endswith("_resets") and row["value"] is not None and row["value"] > 0
    }
    for row in tables["device"]:
        entity, time, field, value = row["entity_id"], row["time"], row["field"], row["value"]
        if row["measurement"] == "bme680":
            sensor_times[entity].add(time)
        elif value is not None:
            if field.endswith("_resets") and value > 0:
                event("device", entity, time, time + timedelta(minutes=1), "counter-reset", field)
            if field in FAILURE_COUNTERS:
                old = previous.get((entity, field))
                if (
                    old
                    and time - old[0] == timedelta(minutes=1)
                    and value > old[1]
                    and (entity, time, field.removesuffix("_last")) not in resets
                ):
                    event(
                        "device",
                        entity,
                        time,
                        time + timedelta(minutes=1),
                        FAILURE_COUNTERS[field],
                        field,
                    )
                previous[(entity, field)] = time, value
    for row in tables["container"]:
        if row["field"] == "health_status_last" and row["text"] == "unhealthy":
            event(
                "container",
                row["entity_id"],
                row["time"],
                row["time"] + timedelta(minutes=5),
                "unhealthy",
                "health_status_last",
            )
    all_times = set().union(*sensor_times.values()) if sensor_times else set()
    for entity, times in [*sorted(sensor_times.items()), ("*", all_times)]:
        for time in sorted(times):
            prior = last_seen.get(entity)
            if prior and time - prior >= timedelta(minutes=11):
                note = "Absence of sensor telemetry; does not establish the cause"
                if entity == "*" and any(
                    prior.date().isoformat() <= day <= time.date().isoformat()
                    for day in KNOWN_EVENT_DATES
                ):
                    note += "; overlaps a candidate event date in GitHub #18 (unreviewed)"
                event(
                    "device",
                    entity,
                    prior + timedelta(minutes=1),
                    time,
                    "all-devices-silent" if entity == "*" else "sensor-telemetry-gap",
                    note,
                )
            last_seen[entity] = time
    return events


def build_dataset(
    client: httpx.Client,
    output: Path,
    start: datetime,
    stop: datetime,
    train_stop: datetime,
    validation_stop: datetime,
    manual_path: Path | None,
) -> dict[str, Any]:
    if output.exists():
        raise ValueError(f"Output already exists; choose a new dataset directory: {output}")
    if not start < train_stop < validation_stop < stop:
        raise ValueError("Require start < train-stop < validation-stop < stop")
    manual = load_manual_labels(manual_path, train_stop, validation_stop)
    manual_matches: set[int] = set()
    previous: dict[tuple[str, str], tuple[datetime, float]] = {}
    last_seen: dict[str, datetime] = {}
    coverage: dict[str, dict[str, Any]] = {kind: {} for kind in KINDS}
    totals: dict[str, Counter[str]] = {kind: Counter() for kind in KINDS}
    templates: dict[tuple[str, str], dict[str, Any]] = {}
    source_hash = hashlib.sha256()
    output.parent.mkdir(parents=True, exist_ok=True)
    with (
        TemporaryDirectory(prefix=".advisor-", dir=output.parent) as temporary,
        ExitStack() as stack,
    ):
        staging = Path(temporary) / "dataset"
        staging.mkdir()
        writers = {
            kind: stack.enter_context(pq.ParquetWriter(staging / f"{kind}.parquet", SCHEMA))
            for kind in KINDS
        }
        labels_file = stack.enter_context(
            (staging / "weak-labels.csv").open("w", encoding="utf-8", newline="")
        )
        label_writer = csv.DictWriter(labels_file, fieldnames=EVENT_COLUMNS)
        label_writer.writeheader()
        event_counts: Counter[str] = Counter()
        cursor = start
        while cursor < stop:
            end = min(cursor + timedelta(hours=6), stop)
            tables: dict[str, list[dict[str, Any]]] = {kind: [] for kind in KINDS}
            for bucket in ("sensors_1m", "metrics_5m"):
                normalized = normalize(
                    read_records(client, bucket, cursor, end), bucket, cursor, end
                )
                for kind, rows in normalized.items():
                    tables[kind].extend(rows)
            events = weak_events(tables, previous, last_seen)
            for event in events:
                label_writer.writerow(event)
                event_counts[event["label"]] += 1
                for index, label in enumerate(manual):
                    if (
                        label["entity_type"] == event["entity_type"]
                        and label["entity_id"] == event["entity_id"]
                        and event["start"] <= label["start"] < label["stop"] <= event["stop"]
                    ):
                        manual_matches.add(index)
                begin, finish = max(event["start"], train_stop), min(event["stop"], validation_stop)
                if begin < finish:
                    templates.setdefault(
                        (event["entity_type"], event["entity_id"]),
                        {
                            "entity_type": event["entity_type"],
                            "entity_id": event["entity_id"],
                            "start": begin.isoformat(),
                            "stop": finish.isoformat(),
                            "label": "",
                            "note": "",
                        },
                    )
            positive: dict[tuple[str, str, datetime], set[str]] = defaultdict(set)
            for event in events:
                if event["label"] not in {"sensor-telemetry-gap", "all-devices-silent"}:
                    positive[(event["entity_type"], event["entity_id"], event["start"])].add(
                        event["label"]
                    )
            for kind, rows in tables.items():
                windows: dict[str, set[datetime]] = defaultdict(set)
                for row in rows:
                    time, entity = row["time"], row["entity_id"]
                    source_hash.update(
                        json.dumps(row, sort_keys=True, default=str).encode() + b"\n"
                    )
                    row["split"] = (
                        "train"
                        if time < train_stop
                        else "validation"
                        if time < validation_stop
                        else "test"
                    )
                    row["weak_labels"] = sorted(positive.get((kind, entity, time), ()))
                    row["manual_label"] = None
                    for index, label in enumerate(manual):
                        if (
                            label["entity_type"] == kind
                            and label["entity_id"] == entity
                            and label["start"] <= time < label["stop"]
                        ):
                            if time + timedelta(minutes=row["interval_minutes"]) > label["stop"]:
                                raise ValueError(
                                    "Manual label boundaries must cover entire aggregates"
                                )
                            row["manual_label"] = label["label"]
                            manual_matches.add(index)
                    windows[entity].add(time)
                    totals[kind][row["split"]] += 1
                    if row["split"] == "validation":
                        templates.setdefault(
                            (kind, entity),
                            {
                                "entity_type": kind,
                                "entity_id": entity,
                                "start": time.isoformat(),
                                "stop": (
                                    time + timedelta(minutes=row["interval_minutes"])
                                ).isoformat(),
                                "label": "",
                                "note": "",
                            },
                        )
                for entity, times in windows.items():
                    info = coverage[kind].setdefault(
                        entity, {"first": min(times), "last": max(times), "observed_windows": 0}
                    )
                    info["last"] = max(times)
                    info["observed_windows"] += len(times)
                if rows:
                    writers[kind].write_table(pa.Table.from_pylist(rows, schema=SCHEMA))
            print(f"Exported through {end.isoformat()}", flush=True)
            cursor = end
        if not totals["device"]:
            raise ValueError("No device aggregates found in the requested period")
        if len(manual_matches) != len(manual):
            raise ValueError("Some manual label intervals matched no aggregate rows")
        for kind, entities in coverage.items():
            cadence = 5 if kind in {"host", "container"} else 1
            for info in entities.values():
                expected = int((info["last"] - info["first"]).total_seconds() / (60 * cadence)) + 1
                info["missing_windows"] = expected - info["observed_windows"]
                info["first"], info["last"] = info["first"].isoformat(), info["last"].isoformat()
        for writer in writers.values():
            writer.close()
        labels_file.close()
        with (staging / "manual-labels-template.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer_csv = csv.DictWriter(handle, fieldnames=MANUAL_COLUMNS)
            writer_csv.writeheader()
            writer_csv.writerows(templates[key] for key in sorted(templates)[:20])
        with (staging / "manual-labels.csv").open("w", encoding="utf-8", newline="") as handle:
            writer_csv = csv.DictWriter(handle, fieldnames=EVENT_COLUMNS, extrasaction="ignore")
            writer_csv.writeheader()
            writer_csv.writerows(manual)
        parquet_hashes = {}
        for kind in KINDS:
            with (staging / f"{kind}.parquet").open("rb") as handle:
                parquet_hashes[kind] = hashlib.file_digest(handle, "sha256").hexdigest()
        card = {
            "schema_version": "advisor-1",
            "start": start.isoformat(),
            "stop": stop.isoformat(),
            "split": {
                "train_stop": train_stop.isoformat(),
                "validation_stop": validation_stop.isoformat(),
                "shuffle": False,
            },
            "sources": {"sensors_1m": 1, "metrics_5m": 5},
            "source_values_sha256": source_hash.hexdigest(),
            "known_event_dates_unreviewed": KNOWN_EVENT_DATES,
            "weak_label_intervals": dict(sorted(event_counts.items())),
            "manual_labels": {
                "reviewed_intervals": len(manual),
                "input_sha256": hashlib.sha256(manual_path.read_bytes()).hexdigest()
                if manual_path
                else None,
            },
            "tables": {
                kind: {
                    "rows": sum(totals[kind].values()),
                    "splits": dict(totals[kind]),
                    "entities": coverage[kind],
                    "sha256": parquet_hashes[kind],
                }
                for kind in KINDS
            },
        }
        write_report(card, staging / "data-card.json")
        (staging / "data-card.md").write_text(
            "# Advisor dataset\n\n"
            f"UTC coverage requested: [{start.isoformat()}, {stop.isoformat()}).\n\n"
            "Four long-form entity tables; numeric `value` and string/boolean `text` "
            "remain separate. Sensors/rooms use 1 minute; host/container history retains "
            "its actual 5-minute cadence. Window-start timestamps and half-open "
            "time-based splits are shared by every entity.\n\n"
            "No interpolation, forward filling, normalization or assumed-normal labels. "
            "Telemetry gaps of at least 10 missing minutes appear only in weak-labels.csv, "
            "including `entity_id=*` for all-device sensor silence. "
            "Initial/final periods outside observed entity coverage are unknown. "
            "A gap does not identify its cause. Event dates in #18 are unreviewed hints; "
            "the unknown timezone prevents treating the quoted hours as ground truth.\n\n"
            "Weak positives: adjacent-minute failure-counter increases without resets, "
            "observed counter resets, and explicit unhealthy container status. "
            "Counter changes across gaps/resets are not failure rates. "
            "Grafana alert history is not imported. "
            "Manual labels and weak labels stay separate.\n\n"
            "Rooms use historical BME680 room tags: "
            "sample-weighted means, min/max and summed counts. "
            "Status is joined by device ID only, never assigned today's room. "
            "Metrics retain source tag dimensions, including container state/version: "
            "different partial aggregates in one window remain distinct under one entity ID. "
            "Test devices, Docker memory, IP/SSID fields and unsupported Hue measurements "
            "are excluded. IAQ/quality values retain source semantics; "
            "consumers must inspect warmed_up/time_valid. "
            "Boot IDs and sequence numbers are metadata, not climate features.\n\n"
            f"Reviewed manual intervals: {len(manual)}. Complete manual-labels-template.csv from "
            "raw observations/incident evidence and rebuild with --manual-labels. Blank labels are "
            "unreviewed. Use only validation intervals for this small reviewed set.\n\n"
            "data-card.json records per-entity coverage/gaps, splits, label counts and hashes. "
            "Rerun identical boundaries against unchanged aggregate history for reproducibility; "
            "source late writes change the input hash. Build forecasting windows within one split, "
            "keep targets inside that split, and fit preprocessing on training data only.\n",
            encoding="utf-8",
        )
        staging.rename(output)
    return card


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("start", "train-stop", "validation-stop", "stop"):
        parser.add_argument(f"--{name}", type=boundary, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manual-labels", type=Path)
    args = parser.parse_args(argv)
    try:
        url, token = os.environ.get("INFLUX_URL", ""), os.environ.get("INFLUX_TOKEN", "")
        if not url.startswith(("https://", "http://")) or not token:
            raise ValueError("Set INFLUX_URL and INFLUX_TOKEN; INFLUX_ORG defaults to smarthome")
        with httpx.Client(
            base_url=url.rstrip("/"),
            timeout=60,
            params={"org": os.environ.get("INFLUX_ORG", "smarthome")},
            headers={"Authorization": f"Token {token}", "Accept": "application/csv"},
        ) as client:
            card = build_dataset(
                client,
                args.output_dir,
                args.start,
                args.stop,
                args.train_stop,
                args.validation_stop,
                args.manual_labels,
            )
        print(
            f"Dataset: {args.output_dir} ({sum(t['rows'] for t in card['tables'].values())} rows)"
        )
        return 0
    except (httpx.HTTPError, OSError, ValueError, pa.ArrowException) as error:
        print(f"Advisor dataset build failed: {error}")
        return 1
