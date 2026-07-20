from typing import Any
from urllib.parse import quote

import httpx2

from edgeops_ai.schemas import DeviceObservation, ObservationV1

METRIC_MAPPING = {
    "ingest_messages_enqueued_total": "messages_enqueued_total",
    "ingest_messages_processed_total": "messages_processed_total",
    "ingest_queue_full_total": "queue_full_total",
    "ingest_transform_success_total": "transform_success_total",
    "ingest_transform_failed_total": "transform_failed_total",
    "influx_lines_written_total": "influx_lines_written_total",
    "ingest_pipeline_duration_seconds_sum": ("pipeline_duration_seconds_sum"),
    "ingest_pipeline_duration_seconds_count": ("pipeline_duration_seconds_count"),
}


class CollectorClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout_seconds: float,
        device_ids: tuple[str, ...] = (),
        transport: httpx2.AsyncBaseTransport | None = None,
    ) -> None:
        self._device_ids = device_ids

        self._client = httpx2.AsyncClient(
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
            for source_name, target_name in METRIC_MAPPING.items()
        }

        devices = [await self._fetch_device(device_id) for device_id in self._device_ids]

        return ObservationV1.model_validate(
            {
                "schema_version": "1.0",
                "metadata": payload["metadata"],
                "metrics": metrics,
                "devices": devices,
            }
        )

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
