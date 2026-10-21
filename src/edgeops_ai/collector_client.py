import asyncio
import logging
from typing import Any
from urllib.parse import quote

import httpx

from edgeops_ai.schemas import (
    METRICS,
    OPTIONAL_METRIC_MAPPING,
    REQUIRED_METRIC_MAPPING,
    ZERO_DEFAULT_METRIC_MAPPING,
    DeviceObservation,
    ObservationV1,
)

logger = logging.getLogger(__name__)

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

        # A successful production metrics response comes from a fresh ingest scrape.
        metadata = payload["metadata"]
        if metadata.get("collector_mode") == "production" and metadata.get("simulated") is False:
            collector_metrics = dict(collector_metrics)
            for metric in METRICS.values():
                if metric.absent_zero:
                    collector_metrics.setdefault(metric.source_name, 0.0)

        metrics = {
            target_name: collector_metrics[source_name]
            for source_name, target_name in REQUIRED_METRIC_MAPPING.items()
        }
        metrics.update(
            {
                target_name: collector_metrics.get(source_name) or 0.0
                for source_name, target_name in ZERO_DEFAULT_METRIC_MAPPING.items()
            }
        )
        metrics.update(
            {
                target_name: collector_metrics.get(source_name)
                for source_name, target_name in OPTIONAL_METRIC_MAPPING.items()
            }
        )

        device_tasks = [
            asyncio.create_task(self._fetch_device(device_id)) for device_id in self._device_ids
        ]
        try:
            devices = await asyncio.gather(*device_tasks)
        finally:
            # Drain requests before the next poll or client shutdown, including on failure.
            for task in device_tasks:
                task.cancel()
            await asyncio.gather(*device_tasks, return_exceptions=True)

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
