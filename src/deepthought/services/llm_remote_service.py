from __future__ import annotations

import logging

from nats.aio.client import Client as NATS
from nats.js.client import JetStreamContext

from ..modules.llm_remote import RemoteLLM

logger = logging.getLogger(__name__)


class LLMRemoteService:
    """Service wrapper for :class:`RemoteLLM`."""

    def __init__(self, nats_client: NATS, js_context: JetStreamContext) -> None:
        self._llm = RemoteLLM(nats_client, js_context)

    async def start(self, durable_name: str = "llm_remote_service") -> bool:
        return await self._llm.start_listening(durable_name=durable_name)

    async def stop(self) -> None:
        await self._llm.stop_listening()
