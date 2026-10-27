import asyncio
import hashlib
import json
from dataclasses import replace

from fastapi.testclient import TestClient

from edgeops_ai.advisor import DAY, MINUTE, DeviceSample, PipelineSample, day_start
from edgeops_ai.main import create_app
from edgeops_ai.maintenance import summarize_day
from edgeops_ai.resources import STEP, ResourceSample
from edgeops_ai.schemas import SourceIdentity
from edgeops_ai.state_report import StateReport
from test_state_report import FixtureReader, SequenceDetector, settings

HISTORICAL_ROOM = 'Küche (alt), "Nord" / 測'
HOSTS = ['pi/ä, "eins"', 'pi/ü, "zwei"']
CONTAINER = 'ingest/ä, "gleich"'


class IdentityReader(FixtureReader):
    async def read(self, start, stop):
        window = await super().read(start, stop)
        end = stop.replace(minute=stop.minute - stop.minute % 5)
        resources, pipeline, history = [], [], []
        for host in HOSTS:
            for measurement, metric, value, tags in (
                ("cpu", "usage_active_mean", 95, {"cpu": "cpu-total"}),
                ("mem", "used_percent_mean", 95, {}),
                ("disk", "used_percent_mean", 95, {"path": "/"}),
                (
                    "docker_container_health",
                    "health_status_last",
                    "unhealthy",
                    {"container_name": CONTAINER},
                ),
            ):
                dimensions = {"host": host, **tags}
                entity = json.dumps(dimensions, sort_keys=True)
                resources.extend(
                    ResourceSample(
                        end - (3 - i) * STEP, entity, measurement, metric, value, dimensions
                    )
                    for i in range(3)
                )
            dimensions = {"host": host, "container_name": CONTAINER}
            entity = json.dumps({"measurement": "prometheus", **dimensions}, sort_keys=True)
            pipeline.extend(
                PipelineSample(
                    stop - (2 - i) * MINUTE,
                    entity,
                    "errors",
                    i,
                    source=SourceIdentity(
                        kind="pipeline",
                        measurement="prometheus",
                        host=host,
                        container_name=CONTAINER,
                        dimensions=dimensions,
                    ),
                )
                for i in range(2)
            )
        # An opaque entity must not be parsed to invent a host or container name.
        resources.append(
            ResourceSample(
                end - STEP,
                "pretend-host/pretend-container",
                "docker_container_health",
                "health_status_last",
                "unhealthy",
            )
        )
        base = day_start(stop) - 9 * DAY
        for index in range(9):
            at = base + index * DAY
            devices = [
                DeviceSample(
                    at + minute * MINUTE,
                    "sensor",
                    {
                        "uptime": index * 86400 + minute * 60 + 1000,
                        "free_mem": 100_000 - index * 1000,
                    },
                    HISTORICAL_ROOM,
                )
                for minute in range(720)
            ]
            disks = [
                ResourceSample(
                    at + i * STEP,
                    json.dumps({"host": host, "path": "/"}, sort_keys=True),
                    "disk",
                    "used_percent_mean",
                    70 + index,
                    {"host": host, "path": "/"},
                )
                for host in HOSTS
                for i in range(144)
            ]
            history.extend(summarize_day(devices, disks, at))
        return replace(
            window,
            pipeline=pipeline,
            resources=tuple(resources),
            history=tuple(history),
            history_start=base,
        )


def test_endpoint_preserves_sources_and_matching_estimates_without_parsing_labels():
    with TestClient(create_app(settings(), advisor_reader=IdentityReader())) as client:
        response = client.get("/v1/state-report")
        assert response.status_code == 200, response.text
        report = response.json()
        findings = report["findings"]
        for code, kind in (
            ("host.cpu-activity", "host"),
            ("host.mem-pressure", "host"),
            ("host.disk-pressure", "disk"),
            ("pipeline.errors", "pipeline"),
        ):
            items = [row for row in findings if row["code"] == code]
            assert len(items) == 2
            assert {row["source"]["host"] for row in items} == set(HOSTS)
            assert {row["source"]["kind"] for row in items} == {kind}
            assert len({row["id"] for row in items}) == 2
        containers = [row for row in findings if row["code"] == "host.container-health"]
        assert len(containers) == 3
        known = [row for row in containers if row["source"]["host"] is not None]
        assert {row["source"]["container_name"] for row in known} == {CONTAINER}
        assert {row["source"]["host"] for row in known} == set(HOSTS)
        unknown = next(row for row in containers if row["source"]["host"] is None)
        assert unknown["source"]["container_name"] is None
        assert unknown["subject"] == "pretend-host/pretend-container"
        for estimate in report["maintenance_estimates"]:
            source = estimate["source"]
            assert estimate["advice_due"] is True
            finding = next(
                row
                for row in findings
                if row["code"] == f"maintenance.{estimate['metric']}" and row["source"] == source
            )
            digest = hashlib.sha256(
                f"{estimate['entity_id']}\ntrend:theil-sen@1".encode()
            ).hexdigest()[:16]
            assert finding["id"] == f"{finding['code']}.{digest}"
            if source["kind"] == "device":
                assert source["device_id"] == "sensor"
                assert source["room"] == HISTORICAL_ROOM
                decline = next(row for row in findings if row["code"] == "device.memory-decline")
                assert decline["source"] == source
            else:
                assert source["kind"] == "disk"
                assert source["host"] in HOSTS
                assert source["path"] == "/"
                disk = next(
                    row
                    for row in findings
                    if row["code"] == "host.disk-pressure"
                    and row["source"]["host"] == source["host"]
                )
                assert disk["source"] == source
        # Current observations have a different room; they cannot relabel history.
        current = next(row for row in findings if row["code"] == "device.sensor-read-failures")
        assert current["source"]["room"] == "kitchen"
        schema = client.get("/openapi.json").json()["components"]["schemas"]
        assert "SourceIdentity" in schema
        assert "source" in schema["Finding"]["properties"]
        assert "source" in schema["MaintenanceEstimate"]["properties"]
        assert "code" in schema["Finding"]["properties"]
        # Optional additions keep older schema-1.0 payloads valid.
        for row in report["findings"]:
            row.pop("source")
            row.pop("code")
        for row in report["maintenance_estimates"]:
            row.pop("source")
        assert StateReport.model_validate(report).schema_version == "1.0"


def test_device_silence_uses_room_before_the_gap_and_duplicate_room_labels_keep_device_ids():
    class Reader(FixtureReader):
        async def read(self, start, stop):
            window = await super().read(start, stop)
            devices = [
                replace(row, device_id=device, room=HISTORICAL_ROOM)
                for device in ("sensor/ä", "sensor/ü")
                for row in window.devices
            ]
            devices.extend(
                [
                    DeviceSample(
                        stop - 25 * MINUTE, "silent/ä", {"sensor_samples": 1}, HISTORICAL_ROOM
                    ),
                    DeviceSample(stop - MINUTE, "silent/ä", {"free_mem": 1000}, "new room"),
                ]
            )
            return replace(window, devices=devices)

    with TestClient(create_app(settings(), advisor_reader=Reader())) as client:
        response = client.get("/v1/state-report")
        assert response.status_code == 200, response.text
        findings = response.json()["findings"]
        devices = [row for row in findings if row["code"] == "device.sensor-read-failures"]
        assert {row["source"]["device_id"] for row in devices} == {"sensor/ä", "sensor/ü"}
        assert {row["source"]["room"] for row in devices} == {HISTORICAL_ROOM}
        stale = next(row for row in findings if row["code"] == "device.telemetry-stale")
        assert stale["source"]["device_id"] == "silent/ä"
        assert stale["source"]["room"] == HISTORICAL_ROOM
        climate = next(row for row in findings if row["code"] == "climate.mould-risk")
        assert climate["source"]["kind"] == "room"
        assert climate["source"]["device_id"] in {"sensor/ä", "sensor/ü"}
        assert climate["source"]["room"] == HISTORICAL_ROOM


def test_custom_detector_without_metadata_keeps_unknown_source():
    from edgeops_ai.state_report import StateReporter

    report = asyncio.run(StateReporter(FixtureReader(), (SequenceDetector(),)).report())
    assert report.findings[0].source is None
    assert report.findings[0].code == "host.fixture"
