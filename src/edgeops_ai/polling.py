import asyncio
import logging
from contextlib import suppress

import httpx

from edgeops_ai.analysis import Analyzer
from edgeops_ai.collector_client import CollectorClient

logger = logging.getLogger(__name__)


async def poll_collector(
    analyzer: Analyzer, client: CollectorClient, interval_seconds: float
) -> None:
    while True:
        try:
            analyzer.analyze(await client.fetch_observation())
            analyzer.collector_last_error = None
        except asyncio.CancelledError:
            raise
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as error:
            analyzer.collector_last_error = str(error)
            logger.warning("Collector polling failed: %s", error)
        await asyncio.sleep(interval_seconds)


async def stop_polling(task: asyncio.Task[None]) -> None:
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task
