import asyncio
import logging
from contextlib import suppress
from datetime import UTC, datetime

import httpx

from edgeops_ai.analysis import Analyzer
from edgeops_ai.collector_client import CollectorClient

logger = logging.getLogger(__name__)


async def poll_collector(
    analyzer: Analyzer, client: CollectorClient, interval_seconds: float
) -> None:
    while True:
        try:
            await analyzer.analyze(await client.fetch_observation())
            analyzer.collector_last_error = None
            analyzer.collector_last_success_at = datetime.now(UTC)
            analyzer.collector_consecutive_failures = 0
        except asyncio.CancelledError:
            raise
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as error:
            analyzer.collector_last_error = str(error)
            analyzer.collector_consecutive_failures += 1
            analyzer.collector_poll_failures_total += 1
            logger.warning("Collector polling failed: %s", error)
        await asyncio.sleep(interval_seconds)


async def stop_polling(task: asyncio.Task[None]) -> None:
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task
