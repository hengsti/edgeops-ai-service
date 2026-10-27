"""Exercise optional runtime consumers on frozen held-out windows, without fitting or deployment."""

import argparse
import json
from collections import defaultdict
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from time import sleep
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from edgeops_ai.advisor import DAY, MINUTE, DeviceSample
from edgeops_ai.forecast_runtime import checked_json
from edgeops_ai.main import create_app
from edgeops_ai.resources import STEP, ResourceSample
from edgeops_ai.settings import Settings
from edgeops_ai.state_report import AdvisorWindow
from edgeops_ai.training.advisor_forecast import load_candidate


def held_out_samples(windows: dict) -> list[dict]:
    groups: dict = defaultdict(list)
    for split in ("validation", "test"):
        for window in windows[split]:
            groups[split, window["entity_id"], window["room"]].append(window)
    return [
        items[i] for items in groups.values() for i in sorted({0, len(items) // 2, len(items) - 1})
    ]


def report_response(client: TestClient):
    response = client.get("/v1/state-report")
    while response.status_code == 202:
        sleep(int(response.headers["Retry-After"]))
        response = client.get("/v1/state-report")
    return response


def validate(config_path: Path, climate_data: Path, resource_data: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    settings = Settings(
        _env_file=None,
        collector_polling_enabled=False,
        advisor_influx_url=None,
        advisor_influx_token=None,
        advisor_forecast_config=config_path,
        advisor_weather_postcode="",
        advisor_weather_api_key=None,
    )

    class Clock(datetime):
        current = datetime(2026, 1, 1, tzinfo=UTC)

        @classmethod
        def now(cls, tz=None):
            return cls.current

    class Reader:
        window = AdvisorWindow(Clock.current - DAY, Clock.current, [], [])

        async def read(self, start, stop):
            return replace(self.window, start=start, stop=stop)

    reader = Reader()
    result: dict = {
        "schema_version": "advisor-runtime-validation-1",
        "deployment_approved": False,
        "scope": (
            "Sampled held-out endpoint/contract verification; "
            "no new advice-quality acceptance or Pi measurement."
        ),
        "consumers": {},
    }
    with (
        patch("edgeops_ai.state_report.datetime", Clock),
        TestClient(create_app(settings, advisor_reader=reader)) as client,
    ):
        reporter = client.app.state.state_reporter
        for detector in reporter.forecast_detectors:
            if detector.name not in config["consumers"]:
                continue
            if detector.model is None:
                raise ValueError(f"{detector.name}: frozen runtime evidence was rejected")
            entry = config["consumers"][detector.name]
            source = resource_data if detector.name == "resources" else climate_data
            model, windows, hashes = load_candidate(
                source,
                config_path.parent / entry["candidate"],
                one_step=detector.name == "resources",
            )
            cases = []
            for item in held_out_samples(windows):
                times, values = item["times"], item["past"][0]
                if detector.name != "climate":
                    times = times + item["target_times"]
                    values = np.concatenate((values, item["future"][0]))
                step = STEP if detector.name == "resources" else MINUTE
                stop = times[-1] + step
                if detector.name == "resources":
                    devices = []
                    resources = tuple(
                        ResourceSample(
                            at,
                            item["entity_id"],
                            "cpu",
                            "usage_active_mean",
                            float(value[0]),
                            {"host": item["entity_id"], "cpu": "cpu-total"},
                        )
                        for at, value in zip(times, values, strict=True)
                    )
                else:
                    devices = [
                        DeviceSample(
                            at,
                            item["entity_id"],
                            {
                                "temperature_mean": float(value[0]),
                                "humidity_mean": float(value[1]),
                                "warmed_up": "true",
                                "sensor_samples": 1,
                            },
                            item["room"],
                        )
                        for at, value in zip(times, values, strict=True)
                    ]
                    resources = ()
                reader.window = AdvisorWindow(stop - DAY, stop, devices, [], resources=resources)
                Clock.current = stop + MINUTE
                reporter._cached, reporter._expires = None, 0
                reporter._missing, reporter._climate_advice = {}, ()
                response = report_response(client)
                if response.status_code != 200:
                    raise ValueError(f"{detector.name}: endpoint failed on held-out context")
                report = response.json()
                assert report["models"][detector.name]["status"] == "available", report["models"]
                assert all(row["confidence"] is None for row in report["findings"])
                if detector.name == "climate":
                    frame = report["forecasts"][0]
                    expected = model.predict(item["past"][0], item["times"])[0, :61]
                    assert np.array_equal(frame["temperature_c"], expected[:, 0])
                    assert np.array_equal(frame["humidity_percent"], expected[:, 1])
                    assert datetime.fromisoformat(frame["forecast_origin"]) == stop
                elif detector.name == "health":
                    expected = model.predict(item["past"][0], item["times"])[0]
                    for finding in report["findings"]:
                        if finding["detector"] != "model:ttm-residual@1":
                            continue
                        evidence = {row["metric"]: row["value"] for row in finding["evidence"]}
                        target = datetime.fromisoformat(evidence["forecast_target"])
                        index = int((target - item["target_times"][0]) / MINUTE)
                        channel = model.contract["channels"].index(evidence["channel"])
                        assert evidence["expected_value"] == float(expected[index, channel])
                        assert evidence["observed_value"] == float(
                            item["future"][0, index, channel]
                        )
                cases.append(
                    {
                        "split": item["split"],
                        "entity_id": item["entity_id"],
                        "room": item["room"],
                        "context_start": times[0].isoformat(),
                        "observed_at": stop.isoformat(),
                        "http_status": response.status_code,
                        "model": report["models"][detector.name],
                        "findings": [
                            row
                            for row in report["findings"]
                            if row["detector"].startswith("model:")
                        ],
                        "forecasts": report["forecasts"] if detector.name == "climate" else [],
                    }
                )
            assert {case["split"] for case in cases} == {"validation", "test"}
            pi = checked_json(
                config_path.parent / entry["candidate"] / "pi-benchmark.json", entry["pi_sha256"]
            )
            result["consumers"][detector.name] = {
                "cases": cases,
                "candidate_sha256": hashes,
                "evidence_sha256": entry["evidence_sha256"],
                "pi_evidence_sha256": entry["pi_sha256"],
                "reused_pi_p95_ms": pi["latency_ms"]["p95"],
                "reused_pi_peak_rss_bytes": pi["peak_process_rss_bytes"],
                "calibration_sha256": detector.calibration_hash,
            }
    result["passed"] = bool(result["consumers"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--climate-data", type=Path, required=True)
    parser.add_argument("--resource-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new validation report.")
    result = validate(args.config, args.climate_data, args.resource_data)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "cases": {name: len(value["cases"]) for name, value in result["consumers"].items()},
            }
        )
    )


if __name__ == "__main__":
    main()
