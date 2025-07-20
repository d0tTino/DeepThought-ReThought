from __future__ import annotations

from nats.aio.client import Client as NATS
from nats.js.client import JetStreamContext

from ..modules.output_handler import OutputHandler


class OutputHandlerService:
    """Service wrapper for :class:`OutputHandler`."""

    def __init__(self, nats_client: NATS, js_context: JetStreamContext) -> None:
        self._handler = OutputHandler(nats_client, js_context)

    async def start(self, durable_name: str = "output_handler_service") -> bool:
        return await self._handler.start_listening(durable_name=durable_name)

    async def stop(self) -> None:
        await self._handler.stop_listening()

    async def __aenter__(self) -> "OutputHandlerService":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.stop()
