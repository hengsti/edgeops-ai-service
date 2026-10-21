"""Indoor climate advice from observed minute aggregates."""

import math
from collections import defaultdict, deque
from collections.abc import Sequence
from datetime import datetime, timedelta

from edgeops_ai.advisor import MINUTE, Advice, DeviceSample, check_time


def climate_findings(
    samples: list[DeviceSample],
    start: datetime,
    stop: datetime,
    split_boundaries: tuple[datetime, ...] = (),
    *,
    humidity_minutes: int = 120,
    forecast_advice: Sequence[Advice] = (),
) -> list[Advice]:
    for at in (start, stop, *split_boundaries):
        check_time(at)
        if at.second or at.microsecond:
            raise ValueError("Climate boundaries must align to minutes")
    if start >= stop or any(not start < cut < stop for cut in split_boundaries):
        raise ValueError("Climate replay needs increasing time boundaries")
    if type(humidity_minutes) is not int or humidity_minutes < 1:
        raise ValueError("Humidity duration must be a positive number of minutes")
    seen = set()
    for row in samples:
        check_time(row.at)
        if not row.device_id or not start <= row.at < stop or row.at.second or row.at.microsecond:
            raise ValueError("Invalid climate sample time or device identity")
        sample_key = row.device_id, row.at
        if sample_key in seen:
            raise ValueError("Duplicate climate sample")
        seen.add(sample_key)
        for name in ("humidity_mean", "iaq_mean", "gas_mean"):
            if name not in row.values:
                continue
            value = row.values[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or (name == "gas_mean" and value <= 0)
                or (name != "gas_mean" and not 0 <= value <= 100)
            ):
                raise ValueError(f"Invalid climate metric: {name}")

    humidity: dict[tuple[str, str], deque[DeviceSample]] = defaultdict(
        lambda: deque(maxlen=humidity_minutes)
    )
    air: dict[tuple[str, str], deque[DeviceSample]] = defaultdict(lambda: deque(maxlen=15))
    previous: dict[tuple[str, str], datetime] = {}
    latched: set[tuple[str, str, str]] = set()
    last_advice: dict[str, datetime] = {}
    forecasts = {(item.evidence["device_id"], item.observed_at): item for item in forecast_advice}
    findings = []
    for row in sorted(samples, key=lambda row: (row.at, row.device_id)):
        if not row.room:
            continue
        key = row.device_id, row.room
        before = previous.get(key)
        if (
            before is None
            or row.at - before != MINUTE
            or any(before < cut <= row.at for cut in split_boundaries)
        ):
            humidity[key].clear()
            air[key].clear()
        previous[key] = row.at
        if row.values.get("time_valid") == "false":
            humidity[key].clear()
            air[key].clear()
            continue

        rh = row.values.get("humidity_mean")
        if rh is not None and float(rh) <= 55:
            latched.discard((*key, "climate.mould-risk"))
            latched.discard((*key, "climate.humidity-forecast"))
        if rh is not None and float(rh) > 60:
            humidity[key].append(row)
        else:
            humidity[key].clear()

        score = row.values.get("iaq_mean")
        warmed = row.values.get("warmed_up") == "true"
        if warmed and score is not None and float(score) >= 69.8:
            latched.discard((*key, "climate.air-quality"))
        if warmed and score is not None and float(score) <= 59.8:
            air[key].append(row)
        else:
            air[key].clear()

        observed_at = row.at + MINUTE
        for code, window, required in (
            ("climate.mould-risk", humidity[key], humidity_minutes),
            ("climate.air-quality", air[key], 15),
        ):
            if (
                len(window) < required
                or (*key, code) in latched
                or (
                    row.room in last_advice
                    and observed_at - last_advice[row.room] < timedelta(hours=1)
                )
            ):
                continue
            evidence: dict[str, float | str] = {
                "device_id": row.device_id,
                "observed_minutes": len(window),
            }
            if code == "climate.mould-risk":
                evidence.update(
                    humidity_min_percent=min(float(p.values["humidity_mean"]) for p in window),
                    humidity_max_percent=max(float(p.values["humidity_mean"]) for p in window),
                )
                summary = f"{row.room}: sustained humidity above 60% increases mould risk."
                action = (
                    f"Ventilate {row.room} for 10 minutes if outdoor air is drier, or use a "
                    "dehumidifier; check damp or cold surfaces."
                )
            else:
                evidence.update(
                    iaq_min_score=min(float(p.values["iaq_mean"]) for p in window),
                    iaq_max_score=max(float(p.values["iaq_mean"]) for p in window),
                    iaq_change=float(window[-1].values["iaq_mean"])
                    - float(window[0].values["iaq_mean"]),
                )
                if all("gas_mean" in p.values for p in window):
                    evidence.update(
                        gas_start_ohm=float(window[0].values["gas_mean"]),
                        gas_stop_ohm=float(window[-1].values["gas_mean"]),
                        gas_change_ohm=float(window[-1].values["gas_mean"])
                        - float(window[0].values["gas_mean"]),
                    )
                summary = f"{row.room}: the warmed-up sensor reports sustained poor IAQ."
                action = (
                    f"Check cooking, cleaning or other sources in {row.room}; use extraction "
                    "or ventilate for 10 minutes when outdoor air quality permits."
                )
            findings.append(
                Advice(
                    code,
                    row.room,
                    window[0].at,
                    observed_at,
                    observed_at,
                    summary,
                    action,
                    evidence,
                    detector="rule:indoor-climate@1",
                )
            )
            latched.add((*key, code))
            last_advice[row.room] = observed_at
        forecast = forecasts.get((row.device_id, observed_at))
        if (
            forecast is not None
            and forecast.entity_id == row.room
            and (*key, forecast.code) not in latched
            and (
                row.room not in last_advice
                or observed_at - last_advice[row.room] >= timedelta(hours=1)
            )
        ):
            findings.append(forecast)
            latched.add((*key, forecast.code))
            last_advice[row.room] = observed_at
    return findings
