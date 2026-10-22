"""Runtime contracts use small real ONNX graphs, never training mocks."""

import asyncio
import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime

import onnx
import pytest
from fastapi.testclient import TestClient

from edgeops_ai.advisor import MINUTE, DeviceSample
from edgeops_ai.forecasting import ATOL, MANIFEST_KEY, RTOL, forecast_contract
from edgeops_ai.main import create_app
from edgeops_ai.resources import STEP, ResourceSample
from edgeops_ai.settings import Settings
from edgeops_ai.state_report import AdvisorWindow

START = datetime(2026, 10, 1, tzinfo=UTC)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return digest(path)


@pytest.fixture
def bundle(tmp_path):
    consumers = {}
    for name in ("health", "climate", "resources"):
        root = tmp_path / name
        root.mkdir()
        profile = "host-cpu" if name == "resources" else "climate"
        contract = forecast_contract(profile, 16, 96)
        channels = len(contract["channels"])
        values = [20.0] if channels == 1 else [20.0, 50.0]
        manifest = {
            "schema_version": "advisor-onnx-1",
            "model_version": f"{name}-fixture",
            "contract": contract,
            "dataset": {"data-card.json": "a" * 64},
        }
        output = (
            [v for i in range(96) for v in (20.0, 50.0 if i < 10 else 65.0)]
            if name == "climate"
            else values * 96
        )
        tensor = onnx.helper.make_tensor("value", onnx.TensorProto.FLOAT, [1, 96, channels], output)
        graph = onnx.helper.make_graph(
            [onnx.helper.make_node("Constant", [], ["forecast"], value=tensor)],
            name,
            [
                onnx.helper.make_tensor_value_info(
                    "past_values", onnx.TensorProto.FLOAT, [1, 16, channels]
                )
            ],
            [
                onnx.helper.make_tensor_value_info(
                    "forecast", onnx.TensorProto.FLOAT, [1, 96, channels]
                )
            ],
        )
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)])
        model.ir_version = 10
        onnx.helper.set_model_props(model, {MANIFEST_KEY: json.dumps(manifest)})
        onnx.save(model, root / "model.onnx")
        model_hash = digest(root / "model.onnx")
        (root / "model.onnx.sha256").write_text(model_hash)
        write_json(root / "manifest.json", {**manifest, "sha256": model_hash})
        write_json(
            root / "validation-report.json",
            {
                "schema_version": "advisor-validation-1",
                "passed": True,
                "model_version": manifest["model_version"],
                "sources": manifest["dataset"],
                "tolerances": {"atol": ATOL, "rtol": RTOL},
            },
        )
        pi_hash = write_json(
            root / "pi-benchmark.json",
            {
                "schema_version": "advisor-benchmark-1",
                "model_sha256": model_hash,
                "model_version": manifest["model_version"],
                "contract": contract,
                "architecture": "aarch64",
                "parity_passed": True,
                "resource_targets_met": True,
                "latency_ms": {"p95": 12},
                "peak_process_rss_bytes": 80000000,
            },
        )
        metadata = {
            "model_version": manifest["model_version"],
            "contract": contract,
            "candidate_sha256": {
                p.name: digest(p) for p in root.iterdir() if p.name != "pi-benchmark.json"
            },
            "dataset": manifest["dataset"],
            "deployment_approved": False,
        }
        key = {"health": "forecast_residuals", "climate": "forecast", "resources": "resources"}[
            name
        ]
        if name == "health":
            metadata.update(
                score_threshold=1,
                sustained_minutes=15,
                scale_floor=ATOL,
                devices={
                    "sensor:kitchen": {
                        "training_windows": 2,
                        "training_residual_sha256": "b" * 64,
                        "scale_by_horizon_channel": [[1, 1]] * 96,
                    }
                },
            )
        elif name == "resources":
            calibration = {
                "host": {"ttm": {"scale": 1, "training_samples": 100, "residual_sha256": "b" * 64}}
            }
            metadata.update(
                calibration=calibration,
                calibration_sha256=hashlib.sha256(
                    json.dumps(calibration, sort_keys=True).encode()
                ).hexdigest(),
                cpu_threshold_percent=90,
                sustained_minutes=15,
                lead_minutes=5,
            )
        else:
            metadata.update(humidity_threshold_percent=60, lead_minutes=list(range(61)))
        evidence_hash = write_json(root / "evidence.json", {key: metadata})
        consumers[name] = {
            "candidate": name,
            "model_sha256": model_hash,
            "evidence": f"{name}/evidence.json",
            "evidence_sha256": evidence_hash,
            "pi_sha256": pi_hash,
        }
    path = tmp_path / "runtime.json"
    write_json(path, {"schema_version": "advisor-runtime-1", "consumers": consumers})
    return path


def configured(bundle, **changes):
    return Settings(
        _env_file=None,
        collector_polling_enabled=False,
        advisor_forecast_config=bundle,
        **changes,
    )


class Reader:
    calls = 0
    broken = False

    async def read(self, start, stop):
        self.calls += 1
        devices = [
            DeviceSample(
                stop - (48 - i) * MINUTE,
                "sensor",
                {
                    "temperature_mean": 20 if i < 16 else 25,
                    "humidity_mean": 50,
                    "warmed_up": "true",
                    "time_valid": "true",
                    "sensor_samples": 1,
                },
                "kitchen",
            )
            for i in range(48)
        ]
        end = stop.replace(minute=stop.minute - stop.minute % 5)
        resources = []
        for i in range(20):
            at = end - (20 - i) * STEP
            resources.extend(
                [
                    ResourceSample(
                        at,
                        "host",
                        "cpu",
                        "usage_active_mean",
                        95 if i >= 16 else 20,
                        {"host": "host", "cpu": "cpu-total"},
                    ),
                    ResourceSample(
                        at,
                        "host/service",
                        "docker_container_health",
                        "health_status_last",
                        "unhealthy",
                        {"host": "host", "container_name": "service"},
                    ),
                ]
            )
        if self.broken:
            devices = [row for i, row in enumerate(devices) if i != 20]
        return AdvisorWindow(start, stop, devices, [], resources=tuple(resources))


def test_optional_consumers_serve_real_findings_frames_and_cache(bundle):
    reader = Reader()
    with TestClient(create_app(configured(bundle), advisor_reader=reader)) as client:
        response = client.get("/v1/state-report")
        assert response.status_code == 200, response.text
        report = response.json()
        assert {k: v["status"] for k, v in report["models"].items()} == {
            "health": "available",
            "climate": "available",
            "resources": "available",
        }
        codes = {row["detector"] for row in report["findings"]}
        assert {"model:ttm-residual@1", "model:ttm-workload@1"} <= codes
        health = next(
            row for row in report["findings"] if row["detector"] == "model:ttm-residual@1"
        )
        evidence = {row["metric"]: row["value"] for row in health["evidence"]}
        assert evidence["observed_value"] == 25
        assert evidence["expected_value"] == 20
        assert datetime.fromisoformat(evidence["forecast_target"]) == datetime.fromisoformat(
            health["window_start"]
        )
        host = next(row for row in report["findings"] if row["detector"] == "model:ttm-workload@1")
        assert host["severity"] == "warning"
        assert all(row["confidence"] is None for row in report["findings"])
        frame = report["forecasts"][0]
        assert len(frame["target_times"]) == 61
        assert frame["target_times"][0] == frame["forecast_origin"]
        assert frame["uncertainty"] == "uncalibrated"
        assert frame["humidity_percent"] == [50] * 10 + [65] * 51
        assert client.get("/v1/state-report").json() == report
        assert reader.calls == 1


@pytest.mark.parametrize("problem", ["hash", "parity", "pi", "calibration"])
def test_failed_consumer_keeps_rules_and_is_explicitly_unavailable(bundle, problem):
    root = bundle.parent / "health"
    if problem == "hash":
        with (root / "model.onnx").open("ab") as stream:
            stream.write(b"bad")
    elif problem == "parity":
        path = root / "validation-report.json"
        data = json.loads(path.read_text())
        data["passed"] = False
        write_json(path, data)
    elif problem == "pi":
        write_json(root / "pi-benchmark.json", {})
    else:
        path = root / "evidence.json"
        data = json.loads(path.read_text())
        data["forecast_residuals"]["devices"]["sensor:kitchen"]["scale_by_horizon_channel"] = []
        write_json(path, data)
        config = json.loads(bundle.read_text())
        config["consumers"]["health"]["evidence_sha256"] = digest(path)
        write_json(bundle, config)
    with TestClient(create_app(configured(bundle), advisor_reader=Reader())) as client:
        response = client.get("/v1/state-report")
        assert response.status_code == 200, response.text
        report = response.json()
        assert report["models"]["health"]["status"] == "error"
        assert report["overall"] != "ok"
        assert any(row["detector"].startswith("rule:") for row in report["findings"])
        assert not any(row["detector"] == "model:ttm-residual@1" for row in report["findings"])


def test_configuration_path_alone_enables_consumers(bundle):
    settings = Settings(
        _env_file=None, collector_polling_enabled=False, advisor_forecast_config=bundle
    )
    with TestClient(create_app(settings, advisor_reader=Reader())) as client:
        report = client.get("/v1/state-report").json()
        assert {value["status"] for value in report["models"].values()} == {"available"}


def test_default_report_explicitly_lists_disabled_consumers():
    with TestClient(
        create_app(
            Settings(_env_file=None, collector_polling_enabled=False), advisor_reader=Reader()
        )
    ) as client:
        report = client.get("/v1/state-report").json()
        assert {value["status"] for value in report["models"].values()} == {"disabled"}
        assert report["forecasts"] == []


@pytest.mark.parametrize(
    "problem,status",
    [
        ("warmup", "insufficient-history"),
        ("clock", "insufficient-history"),
        ("reset", "insufficient-history"),
        ("room", "uncalibrated"),
        ("stale", "stale"),
        ("missing-channel", "insufficient-history"),
        ("gap", "available"),
    ],
)
def test_health_quality_breaks_persistence_without_imputation(bundle, problem, status):
    from edgeops_ai.forecast_runtime import load_consumers

    window = asyncio.run(Reader().read(START, START + 1440 * MINUTE))
    rows = list(window.devices)
    if problem == "room":
        rows = [replace(row, room="other-room") for row in rows]
    elif problem == "stale":
        rows = rows[:30]
    elif problem == "gap":
        rows.pop(20)
    else:
        name, value = {
            "warmup": ("warmed_up", "false"),
            "clock": ("time_valid", "false"),
            "reset": ("uptime_resets", 1),
            "missing-channel": ("temperature_mean", None),
        }[problem]
        rows = [
            replace(row, values={**row.values, name: value}) if i >= 20 else row
            for i, row in enumerate(rows)
        ]
    detector = load_consumers(bundle)[0]
    findings = detector.detect(replace(window, devices=rows))
    assert findings == []
    assert detector.availability.status == status


def test_cpu_alone_or_another_hosts_unhealthy_service_cannot_produce_model_warning(bundle):
    from edgeops_ai.forecast_runtime import load_consumers

    window = asyncio.run(Reader().read(START, START + 1440 * MINUTE))
    rows = [
        replace(row, dimensions={**row.dimensions, "host": "other-host"})
        if row.measurement.startswith("docker_")
        else row
        for row in window.resources
    ]
    detector = load_consumers(bundle)[2]
    findings = detector.detect(replace(window, resources=tuple(rows)))
    assert findings and all(item.severity == "info" for item in findings)
    assert all(item.evidence["cause"] == "unknown" for item in findings)


def test_model_failure_is_partial_and_preserves_deterministic_findings(bundle, monkeypatch):
    with TestClient(create_app(configured(bundle), advisor_reader=Reader())) as client:
        detector = client.app.state.state_reporter.forecast_detectors[0]

        def fail(*_):
            raise RuntimeError("inference failed")

        monkeypatch.setattr(detector.model, "predict", fail)
        report = client.get("/v1/state-report").json()
        assert report["models"]["health"]["status"] == "error"
        assert report["status"] == "partial" and report["overall"] == "attention"
        assert any(row["detector"].startswith("rule:") for row in report["findings"])


def test_climate_cooldown_survives_refresh_and_keeps_forecast_origin(bundle, monkeypatch):
    from edgeops_ai import state_report

    class Clock(datetime):
        current = START + 1441 * MINUTE

        @classmethod
        def now(cls, tz=None):
            return cls.current

    monkeypatch.setattr(state_report, "datetime", Clock)
    with TestClient(create_app(configured(bundle), advisor_reader=Reader())) as client:
        first = client.get("/v1/state-report").json()
        finding = next(row for row in first["findings"] if row["detector"] == "model:ttm-climate@1")
        Clock.current += 5 * MINUTE
        client.app.state.state_reporter._expires = 0
        second = client.get("/v1/state-report").json()
        repeated = [row for row in second["findings"] if row["detector"] == "model:ttm-climate@1"]
        assert len(repeated) == 1
        assert repeated[0]["window_stop"] == finding["window_stop"]
        assert second["forecasts"][0]["forecast_origin"] != first["forecasts"][0]["forecast_origin"]


def test_enabled_runtime_never_imports_training_frameworks(bundle):
    code = """
import asyncio, sys
from pathlib import Path
from edgeops_ai.main import create_app
from edgeops_ai.settings import Settings
settings = Settings(_env_file=None, collector_polling_enabled=False,
    advisor_influx_url='http://unused', advisor_influx_token='fixture',
    advisor_forecast_config=Path(sys.argv[1]))
async def check():
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        assert all(item.model is not None for item in app.state.state_reporter.forecast_detectors)
        assert app.state.state_reporter.reader.resource_history_hours == 48
        blocked = {'torch', 'tsfm_public', 'pyarrow', 'scipy', 'sklearn', 'edgeops_ai.training'}
        assert not blocked.intersection(sys.modules)
asyncio.run(check())
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(bundle)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_runtime_validation_samples_both_splits_and_each_entity():
    from scripts.validate_advisor_runtime import held_out_samples

    windows = {
        "validation": [{"entity_id": "a", "room": "r", "index": i} for i in range(5)],
        "test": [{"entity_id": "b", "room": "s", "index": i} for i in range(2)],
    }
    samples = held_out_samples(windows)
    assert [(row["entity_id"], row["index"]) for row in samples] == [
        ("a", 0),
        ("a", 2),
        ("a", 4),
        ("b", 0),
        ("b", 1),
    ]


def test_model_findings_use_existing_two_refresh_recovery(bundle):
    class RecoveringReader(Reader):
        recovered = False

        async def read(self, start, stop):
            window = await super().read(start, stop)
            if self.recovered:
                return replace(
                    window,
                    devices=[
                        replace(row, values={**row.values, "temperature_mean": 20})
                        for row in window.devices
                    ],
                )
            return window

    reader = RecoveringReader()
    with TestClient(create_app(configured(bundle), advisor_reader=reader)) as client:
        first = client.get("/v1/state-report").json()
        identity = next(
            row["id"] for row in first["findings"] if row["detector"] == "model:ttm-residual@1"
        )
        reader.recovered = True
        reporter = client.app.state.state_reporter
        reporter._expires = 0
        assert identity in {row["id"] for row in client.get("/v1/state-report").json()["findings"]}
        reporter._expires = 0
        assert identity not in {
            row["id"] for row in client.get("/v1/state-report").json()["findings"]
        }


@pytest.mark.parametrize("hours", [24, 48])
def test_resource_query_preserves_short_windows_and_extends_only_for_model_context(hours):
    import httpx

    from edgeops_ai.influx_reader import HttpInfluxReader

    queries = []

    def handler(request):
        queries.append(json.loads(request.content)["query"])
        return httpx.Response(200, content="")

    async def run():
        reader = HttpInfluxReader(
            "http://influx",
            "org",
            "fixture",
            resource_history_hours=hours,
            transport=httpx.MockTransport(handler),
        )
        try:
            await reader.read(START, START + 6 * MINUTE)
        finally:
            await reader.close()

    asyncio.run(run())
    query = next(
        query for query in queries if '"metrics_5m"' in query and "advisor-history" not in query
    )
    expected = START if hours == 24 else START + 6 * MINUTE - 2 * 1440 * MINUTE
    assert f"range(start: {expected.isoformat()}" in query
