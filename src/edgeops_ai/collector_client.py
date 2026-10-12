import logging
from typing import Any
from urllib.parse import quote

import httpx

from edgeops_ai.schemas import DeviceObservation, ObservationV1

logger = logging.getLogger(__name__)

REQUIRED_METRIC_MAPPING = {
    "ingest_messages_enqueued_total": "messages_enqueued_total",
    "ingest_messages_processed_total": "messages_processed_total",
    "ingest_transform_success_total": "transform_success_total",
    "ingest_transform_failed_total": "transform_failed_total",
    "influx_lines_written_total": "influx_lines_written_total",
    "influx_write_failed_total": "influx_write_failed_total",
    "influxdb_healthy": "influxdb_healthy",
    "ingest_pipeline_duration_seconds_sum": "pipeline_duration_seconds_sum",
    "ingest_pipeline_duration_seconds_count": "pipeline_duration_seconds_count",
}

OPTIONAL_METRIC_MAPPING = {
    "mqtt_messages_received_total": "mqtt_messages_received_total",
    "ingest_event_queue_full_total": "event_queue_full_total",
    "ingest_decode_success_total": "decode_success_total",
    "ingest_incoming_oversized_total": "incoming_oversized_total",
    "ingest_incoming_non_utf8_total": "incoming_non_utf8_total",
    "ingest_incoming_invalid_json_total": "incoming_invalid_json_total",
    "ingest_validate_raw_success_total": "raw_validation_success_total",
    "ingest_validate_raw_ignored_total": "raw_validation_ignored_total",
    "ingest_validate_raw_failed_total": "raw_validation_failed_total",
    "ingest_transform_attempt_total": "transform_attempt_total",
    "ingest_transform_deserialize_failed_total": "transform_deserialize_failed_total",
    "ingest_transform_ignored_total": "transform_ignored_total",
    "ingest_validate_business_success_total": "business_validation_success_total",
    "ingest_validate_business_failed_total": "business_validation_failed_total",
    "ingest_wal_queue_full_total": "wal_queue_full_total",
    "ingest_queue_closed_total": "wal_queue_closed_total",
    "ingest_durability_ack_failed_total": "durability_ack_failed_total",
    "dlq_messages_published_total": "dlq_messages_published_total",
    "dlq_publish_errors_total": "dlq_publish_errors_total",
    "wal_forwarder_committed_total": "wal_committed_total",
    "wal_forwarder_drop_total": "wal_drop_total",
    "wal_forwarder_retry_total": "wal_retry_total",
    "wal_forwarder_commit_retry_total": "wal_commit_retry_total",
    "wal_forwarder_retry_outage_active": "wal_retry_outage_active",
    "ingest_decode_duration_seconds_sum": "decode_duration_seconds_sum",
    "ingest_decode_duration_seconds_count": "decode_duration_seconds_count",
    "ingest_transform_duration_seconds_sum": "transform_duration_seconds_sum",
    "ingest_transform_duration_seconds_count": "transform_duration_seconds_count",
    "ingest_persist_duration_seconds_sum": "persist_duration_seconds_sum",
    "ingest_persist_duration_seconds_count": "persist_duration_seconds_count",
    "influx_write_duration_seconds_sum": "influx_write_duration_seconds_sum",
    "influx_write_duration_seconds_count": "influx_write_duration_seconds_count",
    "influx_write_success_total": "influx_write_success_total",
    "ingest_queue_depth": "queue_depth",
    "ingest_queue_capacity": "queue_capacity",
    "ingest_sensor_messages_processed_total": "sensor_messages_processed_total",
    "ingest_status_messages_processed_total": "status_messages_processed_total",
}

BROKER_DOWN_STATUSES = frozenset({"created", "restarting", "removing", "paused", "exited", "dead"})


class CollectorClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout_seconds: float,
        device_ids: tuple[str, ...] = (),
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._device_ids = device_ids

        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"X-EdgeOps-Key": api_key},
            timeout=timeout_seconds,
            transport=transport,
        )

    async def fetch_observation(self) -> ObservationV1:
        payload = await self._get_json("/v1/metrics/ingestion")
        collector_metrics = payload["data"]

        metrics = {
            target_name: collector_metrics[source_name]
            for source_name, target_name in REQUIRED_METRIC_MAPPING.items()
        }
        # Preserve the legacy pushed-observation field while using the collector's
        # unambiguous event-queue counter. WAL queue pressure remains separate.
        metrics["queue_full_total"] = collector_metrics.get("ingest_event_queue_full_total") or 0.0
        metrics.update(
            {
                target_name: collector_metrics.get(source_name)
                for source_name, target_name in OPTIONAL_METRIC_MAPPING.items()
            }
        )

        devices = [await self._fetch_device(device_id) for device_id in self._device_ids]

        return ObservationV1.model_validate(
            {
                "schema_version": "1.0",
                "metadata": payload["metadata"],
                "metrics": metrics,
                "devices": devices,
                "broker_available": await self._fetch_broker_available(),
            }
        )

    async def _fetch_broker_available(self) -> bool | None:
        for attempt in range(2):
            try:
                service = (await self._get_json("/v1/services/emqx"))["data"]
                status, health = service.get("status"), service.get("health")
            except httpx.HTTPError as error:
                if attempt == 0:
                    continue
                logger.warning(
                    "EMQX broker telemetry request failed after retry: %s: %s",
                    type(error).__name__,
                    error,
                )
                # Broker telemetry is optional evidence; failures mean unknown, not down.
                return None
            except (ValueError, KeyError, TypeError, AttributeError):
                return None
            break

        if status in BROKER_DOWN_STATUSES or (status == "running" and health == "unhealthy"):
            return False
        if status == "running" and health == "healthy":
            return True
        # "unavailable" (Docker inspection failed), "starting" or no healthcheck.
        return None

    async def _fetch_device(self, device_id: str) -> DeviceObservation:
        encoded_device_id = quote(device_id, safe="")

        payload = await self._get_json(f"/v1/devices/{encoded_device_id}")
        device = payload["data"]

        return DeviceObservation.model_validate(
            {
                "device_id": device.get("device_id", device_id),
                "heartbeat_missing": not device["available"],
                "heartbeat_age_seconds": device.get("heartbeat_age_seconds"),
            }
        )

    async def _get_json(self, path: str) -> dict[str, Any]:
        response = await self._client.get(path)
        response.raise_for_status()

        payload = response.json()

        if not isinstance(payload, dict):
            raise ValueError(f"Collector response for {path} must be JSON Object.")

        return payload

    async def close(self) -> None:
        await self._client.aclose()
