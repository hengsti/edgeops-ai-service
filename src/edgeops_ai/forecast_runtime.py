"""Optional, fixed Advisor consumers; frozen evidence is loaded once, never fitted here."""

import hashlib
import json
import logging
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import cast

import numpy as np

from edgeops_ai.advisor import MINUTE, Advice, check_time
from edgeops_ai.forecast_advice import (
    apply_container_context,
    climate_projection,
    findings_for,
    residual_advice,
)
from edgeops_ai.forecasting import ATOL, build_windows, load_artifact
from edgeops_ai.resources import STEP
from edgeops_ai.state_report import AdvisorWindow, Category, ModelAvailability

logger = logging.getLogger(__name__)
CONSUMERS = {"health": "forecast_residuals", "climate": "forecast", "resources": "resources"}


def checked_json(path: Path, digest: str) -> dict:
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("Advisor evidence SHA-256 mismatch")
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError("Advisor evidence must be an object")
    return value


def device_points(window: AdvisorWindow) -> dict:
    aliases = {
        "temperature_mean": "temp_c_mean",
        "humidity_mean": "rel_hum_perc_mean",
        "warmed_up": "warmed_up_last",
        "time_valid": "time_valid_last",
        "uptime": "uptime_last",
        "uptime_resets": "uptime_resets",
    }
    points = {}
    for row in window.devices:
        check_time(row.at)
        key = row.device_id, row.at
        if (
            not window.start <= row.at < window.stop
            or row.at.second
            or row.at.microsecond
            or not row.device_id
            or key in points
        ):
            raise ValueError("Invalid forecast device identity, cadence or duplicate")
        points[key] = {
            "room": row.room,
            **{aliases[name]: value for name, value in row.values.items() if name in aliases},
        }
    return points


def resource_points(window: AdvisorWindow) -> tuple[dict, dict]:
    points: dict = defaultdict(dict)
    containers: dict = defaultdict(dict)
    for row in window.resources:
        check_time(row.at)
        if (
            row.at.minute % 5
            or row.at.second
            or row.at.microsecond
            or row.at + STEP > window.stop
            or row.at < window.stop - 2 * timedelta(days=1)
        ):
            raise ValueError("Invalid forecast resource cadence or window")
        host = row.dimensions.get("host", row.entity_id.split("/", 1)[0])
        if row.measurement == "cpu" and row.metric == "usage_active_mean":
            if row.dimensions.get("cpu") != "cpu-total":
                continue
            point = points[host, row.at]
            if "usage_active_mean" in point:
                raise ValueError("Ambiguous host CPU identity")
            point.update(
                room="",
                usage_active_mean=row.value,
                cpu_percent=row.value,
                source_identity=row.entity_id,
            )
        elif row.measurement == "mem" and row.metric == "used_percent_mean":
            points[host, row.at]["memory_percent"] = row.value
        elif row.measurement in {"docker_container_health", "docker_container_status"}:
            name = row.dimensions.get("container_name", row.entity_id.split("/", 1)[-1])
            point = containers[f"{host}/{name}", row.at]
            if row.metric in point and point[row.metric] != row.value:
                point["conflict"] = True
            point[row.metric] = row.value
            point.setdefault("versions", set()).add(
                tuple(
                    row.dimensions.get(key, "")
                    for key in (
                        "container_version",
                        "container_image",
                        "org.opencontainers.image.revision",
                    )
                )
            )
    apply_container_context(points, containers)
    for point in points.values():
        point.setdefault("room", "")
    return dict(points), containers


class ForecastDetector:
    def __init__(self, name: str):
        self.name = name
        self.category = cast(
            Category, {"health": "device", "climate": "climate", "resources": "host"}[name]
        )
        self.availability = ModelAvailability(status="disabled")
        self.frames: list[dict] = []
        self.model = None
        self.metadata: dict = {}
        self.calibration_hash: str | None = None

    def load(self, root: Path, config: dict, sessions: dict) -> None:
        candidate = (root / config["candidate"]).resolve()
        key = candidate, config["model_sha256"]
        if key not in sessions:
            sessions[key] = load_artifact(candidate, config["model_sha256"])
        model, hashes = sessions[key]
        evidence = checked_json(root / config["evidence"], config["evidence_sha256"])
        metadata = evidence[CONSUMERS[self.name]]
        pi = checked_json(candidate / "pi-benchmark.json", config["pi_sha256"])
        if (
            metadata["contract"] != model.contract
            or metadata["model_version"] != model.manifest["model_version"]
            or metadata["candidate_sha256"] != hashes
            or metadata.get("dataset", model.manifest["dataset"]) != model.manifest["dataset"]
            or any(
                metadata.get("sources", model.manifest["dataset"]).get(name) != digest
                for name, digest in model.manifest["dataset"].items()
            )
            or pi.get("schema_version") != "advisor-benchmark-1"
            or pi.get("model_sha256") != hashes["model.onnx"]
            or pi.get("model_version") != model.manifest["model_version"]
            or pi.get("contract") != model.contract
            or pi.get("architecture") != "aarch64"
            or pi.get("parity_passed") is not True
            or pi.get("resource_targets_met") is not True
            or not 0 < pi["latency_ms"]["p95"] < 50
            or not 0 < pi["peak_process_rss_bytes"] < 100_000_000
        ):
            raise ValueError("Incompatible consumer evaluation or exact-artifact Pi evidence")
        profile = "host-cpu" if self.name == "resources" else "climate"
        if model.contract["profile"] != profile:
            raise ValueError("Wrong forecast consumer profile")
        if self.name == "health":
            if metadata["score_threshold"] != 1 or metadata["sustained_minutes"] != 15:
                raise ValueError("Unsupported health residual policy")
            for calibration in metadata["devices"].values():
                scale = np.asarray(calibration["scale_by_horizon_channel"], dtype=np.float64)
                if (
                    scale.shape != tuple(model.contract["output"]["shape"][1:])
                    or not np.isfinite(scale).all()
                    or np.any(scale < ATOL)
                    or calibration["training_windows"] < 1
                ):
                    raise ValueError("Invalid frozen health calibration")
            self.calibration_hash = hashlib.sha256(
                json.dumps(metadata["devices"], sort_keys=True, allow_nan=False).encode()
            ).hexdigest()
        elif self.name == "climate":
            if (
                model.contract["prediction_length"] < 61
                or metadata["humidity_threshold_percent"] != 60
                or metadata["lead_minutes"] != list(range(61))
            ):
                raise ValueError("Unsupported climate horizon or threshold")
        else:
            if (
                metadata["cpu_threshold_percent"] != 90
                or metadata["sustained_minutes"] != 15
                or metadata["lead_minutes"] != 5
            ):
                raise ValueError("Unsupported workload policy")
            for calibration in metadata["calibration"].values():
                scale = calibration["ttm"]["scale"]
                if (
                    not np.isfinite(scale)
                    or scale < ATOL
                    or calibration["ttm"]["training_samples"] < 1
                ):
                    raise ValueError("Invalid frozen host calibration")
            self.calibration_hash = hashlib.sha256(
                json.dumps(metadata["calibration"], sort_keys=True).encode()
            ).hexdigest()
            if self.calibration_hash != metadata["calibration_sha256"]:
                raise ValueError("Invalid workload calibration digest")
        self.model, self.metadata = model, metadata
        self.availability = ModelAvailability(
            status="insufficient-history",
            model_version=model.manifest["model_version"],
            model_sha256=hashes["model.onnx"],
            calibration_sha256=self.calibration_hash,
        )

    def detect(self, window: AdvisorWindow) -> list[Advice]:
        self.frames = []
        if self.model is None:
            return []
        try:
            items = self._resources(window) if self.name == "resources" else self._devices(window)
            for item in items:
                item.evidence.update(
                    model_sha256=self.availability.model_sha256 or "",
                    forecast_uncertainty="uncalibrated",
                )
            return items
        except Exception as error:
            # Model failure must retain deterministic advice and never establish overall health.
            logger.warning("Advisor %s inference failed (%s)", self.name, type(error).__name__)
            self.frames = []
            self.availability = self.availability.model_copy(update={"status": "error"})
            return []

    def _devices(self, window: AdvisorWindow) -> list[Advice]:
        assert self.model is not None
        points = device_points(window)
        windows = build_windows(
            points,
            self.model.contract,
            partial=self.name == "health",
            context_only=self.name == "climate",
        )["runtime"]
        states = {}
        for (entity, at), point in sorted(points.items()):
            states[f"{entity}:{point['room']}"] = (
                "stale" if at + MINUTE < window.stop else "insufficient-history"
            )
        scores, findings = [], []
        for item in windows:
            identity = f"{item['entity_id']}:{item['room']}"
            if self.name == "health":
                calibration = self.metadata["devices"].get(identity)
                if calibration is None:
                    states[identity] = "uncalibrated"
                    continue
                prediction = self.model.predict(item["past"][0], item["times"])[0]
                for step, at in enumerate(item["target_times"]):
                    for channel, name in enumerate(self.model.contract["channels"]):
                        expected = float(prediction[step, channel])
                        actual = float(item["future"][0, step, channel])
                        scale = calibration["scale_by_horizon_channel"][step][channel]
                        scores.append(
                            {
                                "entity_id": item["entity_id"],
                                "room": item["room"],
                                "split": "runtime",
                                "channel": name,
                                "time": at.isoformat(),
                                "score": abs(actual - expected) / scale,
                                "forecast_origin": (item["times"][-1] + MINUTE).isoformat(),
                                "forecast_target": at.isoformat(),
                                "horizon_step": step + 1,
                                "observed_value": actual,
                                "expected_value": expected,
                                "residual_scale": scale,
                            }
                        )
                if item["target_times"][-1] + MINUTE == window.stop:
                    states[identity] = "available"
            elif item["times"][-1] + MINUTE == window.stop:
                prediction = self.model.predict(item["past"][0], item["times"])[0, :61]
                if np.any((prediction[:, 1] < 0) | (prediction[:, 1] > 100)):
                    raise ValueError("Forecast humidity outside physical bounds")
                item["target_times"] = [window.stop + i * MINUTE for i in range(61)]
                frame, advice = climate_projection(
                    item, prediction, self.model.manifest["model_version"]
                )
                frame.update(
                    model_sha256=self.availability.model_sha256, uncertainty="uncalibrated"
                )
                self.frames.append(frame)
                if advice is not None:
                    findings.append(advice)
                states[identity] = "available"
        if self.name == "health":
            findings = residual_advice(
                scores, self.model.manifest["model_version"], self.calibration_hash or ""
            )
            provenance = {
                (row["entity_id"], row["room"], row["channel"], row["time"]): row for row in scores
            }
            for finding in findings:
                row = provenance[
                    finding.entity_id,
                    finding.evidence["room"],
                    finding.evidence["channel"],
                    finding.start.isoformat(),
                ]
                finding.evidence.update(
                    {
                        name: row[name]
                        for name in (
                            "forecast_origin",
                            "forecast_target",
                            "horizon_step",
                            "observed_value",
                            "expected_value",
                            "residual_scale",
                        )
                    }
                )
        self._status(states)
        return findings

    def _resources(self, window: AdvisorWindow) -> list[Advice]:
        assert self.model is not None
        points, _ = resource_points(window)
        windows = build_windows(points, self.model.contract, one_step=True)["runtime"]
        predictions = {}
        for item in windows:
            at = item["target_times"][0]
            # Current-state evidence needs only the latest sustained 15-minute interval.
            if at < window.stop - 4 * STEP:
                continue
            predicted = float(self.model.predict(item["past"][0], item["times"])[0, 0, 0])
            if not 0 <= predicted <= 100:
                raise ValueError("CPU expectation outside physical bounds")
            predictions[item["entity_id"], at] = predicted, item["times"][0]
        observations, states = [], {}
        end = window.stop.replace(minute=window.stop.minute - window.stop.minute % 5)
        for (host, at), point in sorted(points.items()):
            if "cpu_percent" not in point:
                continue
            expected, begin = predictions.get((host, at), (None, None))
            calibration = self.metadata["calibration"].get(host)
            score = (
                abs(point["cpu_percent"] - expected) / calibration["ttm"]["scale"]
                if expected is not None and calibration
                else None
            )
            states[host] = (
                "stale"
                if at + STEP < end
                else "uncalibrated"
                if calibration is None
                else "available"
                if score is not None
                else "insufficient-history"
            )
            if at < window.stop - 4 * STEP or score is None:
                continue
            observations.append(
                {
                    "entity_id": host,
                    "time": at.isoformat(),
                    "split": "runtime",
                    "cpu_percent": point["cpu_percent"],
                    "memory_percent": point.get("memory_percent"),
                    "expected_cpu_percent": expected,
                    "ttm_score": score,
                    "expectation_origin": at.isoformat(),
                    "context_start": begin.isoformat(),
                    "affected_services": ",".join(sorted(point.get("affected_services", []))),
                    "container_boundary": point.get("container_boundary", False),
                    "host_boundary": False,
                    "activity": "unknown",
                    "model_version": self.model.manifest["model_version"],
                    "model_sha256": self.availability.model_sha256,
                    "calibration_sha256": self.calibration_hash,
                }
            )
        self._status(states)
        return findings_for(observations, "ttm", 90, 15)

    def _status(self, states: dict) -> None:
        status = next(
            (value for value in states.values() if value != "available"),
            "available" if states else "no-data",
        )
        self.availability = self.availability.model_copy(
            update={"status": status, "entities": states}
        )


def load_consumers(path: Path) -> tuple[ForecastDetector, ...]:
    detectors = tuple(ForecastDetector(name) for name in CONSUMERS)
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
        entries = config["consumers"]
        if config["schema_version"] != "advisor-runtime-1" or set(entries) - CONSUMERS.keys():
            raise ValueError("Unsupported runtime consumer configuration")
    except Exception as error:
        logger.warning("Advisor forecast configuration failed (%s)", type(error).__name__)
        for detector in detectors:
            detector.availability = ModelAvailability(status="error")
        return detectors
    sessions: dict = {}
    for detector in detectors:
        if detector.name not in entries:
            continue
        try:
            detector.load(path.parent, entries[detector.name], sessions)
        except Exception as error:
            logger.warning("Advisor %s artifact rejected (%s)", detector.name, type(error).__name__)
            detector.availability = ModelAvailability(status="error")
    return detectors
