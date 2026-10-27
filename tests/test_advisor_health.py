from datetime import UTC, datetime, timedelta

import pytest

from edgeops_ai.advisor import DAY, DeviceSample, PipelineSample, device_findings, pipeline_findings

START = datetime(2026, 9, 1, tzinfo=UTC)
MINUTE = timedelta(minutes=1)


def samples(device="a", minutes=120, **changes):
    return [
        DeviceSample(
            START + i * MINUTE,
            device,
            {
                "sensor_samples": 30,
                "rssi": -50,
                "free_mem": 470_000,
                "uptime": 2000 + i * 60,
                "sensor_read_failures_total": 0,
                "samples_failed_total": 0,
                "mqtt_connects_total": 1,
                "pressure_hpa": 1000,
                "iaq_mean": 90,
                "iaq_min": 90,
                "warmed_up": "true",
                **changes,
            },
            "kitchen",
        )
        for i in range(minutes)
    ]


def codes(rows, days=1):
    return {f.code for f in device_findings(rows, START, START + timedelta(days=days))}


def test_healthy_device_has_no_advice():
    rows = samples(minutes=1440)
    assert device_findings(rows, START, START + timedelta(days=1)) == []


def test_all_device_silence_is_one_global_finding_with_causal_detection_time():
    rows = [*samples("a", 1), *samples("b", 1)]
    findings = device_findings(rows, START, START + timedelta(minutes=20))
    assert len(findings) == 1
    finding = findings[0]
    assert finding.code == "device.all-silent"
    assert finding.entity_id == "*"
    assert finding.start == START + MINUTE
    assert finding.observed_at == START + 11 * MINUTE
    assert "broker" in finding.recommendation
    assert "cause" in finding.summary.lower()


def test_staleness_does_not_warn_before_first_sample_or_for_short_gaps():
    rows = samples(minutes=1)
    assert device_findings(rows, START - timedelta(days=1), START + 10 * MINUTE) == []


def test_individual_staleness_does_not_claim_firmware_or_broker_failure():
    findings = device_findings([*samples("a", 1), *samples("b", 30)], START, START + 30 * MINUTE)
    assert [(f.code, f.entity_id) for f in findings] == [("device.telemetry-stale", "a")]
    assert "check" in findings[0].recommendation.lower()


def test_future_device_does_not_change_an_earlier_outage():
    first = [*samples("a", 1), DeviceSample(START + 30 * MINUTE, "a", {"sensor_samples": 1})]
    later = [DeviceSample(START + 35 * MINUTE, "b", {"sensor_samples": 1})]
    before = device_findings(first, START, START + 31 * MINUTE)[0]
    after = device_findings(first + later, START, START + 36 * MINUTE)[0]
    assert (before.code, before.start, before.stop, before.evidence) == (
        after.code,
        after.start,
        after.stop,
        after.evidence,
    )


def test_sensor_counter_increase_names_the_device_and_measured_delta():
    rows = samples()
    rows[1].values["sensor_read_failures_total"] = 3
    for row in rows[2:]:
        row.values["sensor_read_failures_total"] = 3
    finding = next(
        f
        for f in device_findings(rows, START, START + 120 * MINUTE)
        if f.code == "device.sensor-read-failures"
    )
    assert finding.evidence["increase"] == 3
    assert "I2C" in finding.recommendation


@pytest.mark.parametrize("reset", ["counter", "uptime", "gap", "within-window"])
def test_counter_resets_and_gaps_are_not_failures(reset):
    rows = samples(minutes=2, sensor_read_failures_total=20)
    if reset == "counter":
        rows[1].values["sensor_read_failures_total"] = 1
    elif reset == "uptime":
        rows[1].values.update(uptime=1, sensor_read_failures_total=100)
    elif reset == "within-window":
        rows[1].values.update(sensor_read_failures_total_resets=1, sensor_read_failures_total=100)
    else:
        rows[1] = DeviceSample(
            START + 3 * MINUTE, "a", {**rows[1].values, "sensor_read_failures_total": 100}
        )
    assert "device.sensor-read-failures" not in codes(rows)


def test_persistent_wifi_and_delivery_counter_trends_are_advice():
    rows = samples(rssi=-80)
    rows[-1].values.update(mqtt_connects_total=4, samples_failed_total=2)
    found = codes(rows)
    assert {"device.weak-wifi", "device.delivery-failures", "device.reconnects"} <= found
    rows = samples()
    rows[0].values["rssi"] = -90
    assert "device.weak-wifi" not in codes(rows)


def test_memory_decline_requires_multiple_days_within_a_continuous_boot():
    rows = []
    for day in range(9):
        for i in range(1440):
            minute = day * 1440 + i
            rows.append(
                DeviceSample(
                    START + minute * MINUTE,
                    "a",
                    {
                        "sensor_samples": 1,
                        "uptime": 2000 + minute * 60,
                        "free_mem": 470_000 - day * 1000,
                    },
                )
            )
    findings = device_findings(rows, START, START + timedelta(days=9))
    memory = [f for f in findings if f.code == "device.memory-decline"]
    assert memory and memory[0].evidence["days"] >= 7
    assert "suspected" in memory[0].summary.lower()
    for row in rows[4 * 1440 :]:
        row.values["uptime"] = 2000 + (row.at - START - timedelta(days=4)).total_seconds()
    assert "device.memory-decline" not in codes(rows, 9)


def test_pressure_offset_uses_simultaneous_other_devices_and_persistence():
    rows = [*samples("a", pressure_hpa=1001), *samples("b"), *samples("c"), *samples("d")]
    findings = device_findings(rows, START, START + 120 * MINUTE)
    pressure = [f for f in findings if f.code == "device.pressure-offset"]
    assert len(pressure) == 1 and pressure[0].entity_id == "a"
    assert pressure[0].evidence["offset_hpa"] == 1
    assert "elevation" in pressure[0].recommendation
    assert "device.pressure-offset" not in codes(samples("a", pressure_hpa=1001))


def test_one_outlier_does_not_implicate_two_healthy_pressure_peers():
    rows = [*samples("a", pressure_hpa=1002), *samples("b"), *samples("c")]
    findings = device_findings(rows, START, START + 120 * MINUTE)
    assert [f.entity_id for f in findings if f.code == "device.pressure-offset"] == []


def test_climate_only_minute_does_not_hide_a_memory_reset():
    rows = []
    for minute in range(9 * 1440):
        uptime = 2000 + (minute if minute < 4 * 1440 else minute - 4 * 1440) * 60
        values = {
            "sensor_samples": 1,
            "uptime": uptime,
            "free_mem": 470000 - (minute // 1440) * 1000,
        }
        if minute == 4 * 1440:
            values = {"sensor_samples": 1}
        rows.append(DeviceSample(START + minute * MINUTE, "a", values))
    assert "device.memory-decline" not in codes(rows, 9)


def test_memory_decline_room_belongs_to_the_contributing_boot_before_a_move():
    rows = [
        DeviceSample(
            START + day * DAY + minute * MINUTE,
            "sensor",
            {"uptime": day * 86400 + minute * 60 + 1000, "free_mem": 100_000 - day * 1000},
            "old room",
        )
        for day in range(7)
        for minute in range(60)
    ]
    rows.append(
        DeviceSample(
            START + 6 * DAY + 120 * MINUTE, "sensor", {"uptime": 5, "free_mem": 200_000}, "new room"
        )
    )
    decline = next(
        row
        for row in device_findings(rows, START, START + 7 * DAY)
        if row.code == "device.memory-decline"
    )
    assert decline.source.device_id == "sensor"
    assert decline.source.room == "old room"


def test_iaq_warmup_is_flagged_without_claiming_bad_room_air_or_single_raw_sample():
    rows = samples(iaq_min=40, iaq_mean=90, warmed_up="false")
    findings = device_findings(rows, START, START + 120 * MINUTE)
    quality = next(f for f in findings if f.code == "device.iaq-quality")
    assert quality.severity == "info"
    assert "warm" in quality.recommendation.lower()
    assert "single sample" not in quality.summary.lower()
    assert "device.iaq-quality" not in codes(samples(iaq_min=40, iaq_mean=40))


def test_iaq_uses_ingestion_score_scale_and_preserves_suppressed_warmup_flags():
    assert "device.iaq-quality" in codes(samples(iaq_min=51, iaq_mean=85))
    rows = samples(warmed_up="false")
    for row in rows:
        row.values.pop("iaq_min")
        row.values.pop("iaq_mean")
    quality = next(
        f
        for f in device_findings(rows, START, START + 120 * MINUTE)
        if f.code == "device.iaq-quality"
    )
    assert "min_iaq" not in quality.evidence


def test_pipeline_detects_deltas_without_counting_resets_gaps_or_initial_totals():
    def point(minute, value):
        return PipelineSample(START + minute * MINUTE, "ingest", "invalid_json", value)

    findings = pipeline_findings([point(0, 100), point(1, 102), point(2, 1), point(10, 99)])
    assert len(findings) == 1
    assert findings[0].evidence["invalid_json"] == 2
    assert "DLQ" in findings[0].recommendation
    assert pipeline_findings([point(0, 99)]) == []


def test_non_finite_and_naive_input_is_rejected():
    with pytest.raises(ValueError):
        device_findings(samples(rssi=float("nan")), START, START + 120 * MINUTE)
    with pytest.raises(ValueError):
        pipeline_findings([PipelineSample(START.replace(tzinfo=None), "ingest", "invalid_json", 1)])


def test_daily_counter_advice_is_clipped_at_shared_split_boundaries():
    rows = samples(minutes=1440)
    for row in rows[60:]:
        row.values["sensor_read_failures_total"] = 3
    cut = START + timedelta(hours=12)
    findings = device_findings(rows, START, START + timedelta(days=1), (cut,))
    failure = next(f for f in findings if f.code == "device.sensor-read-failures")
    assert failure.start == START and failure.stop == cut
    assert failure.observed_at == cut
