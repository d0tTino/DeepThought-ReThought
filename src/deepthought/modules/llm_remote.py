"""HTTP-based LLM module for the demo."""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

import aiohttp
import nats
from nats.aio.client import Client as NATS
from nats.aio.msg import Msg
from nats.js.client import JetStreamContext

from ..eda.events import EventSubjects, ResponseGeneratedPayload
from ..eda.publisher import Publisher
from ..eda.subscriber import Subscriber

logger = logging.getLogger(__name__)


class RemoteLLM:
    """LLM module that calls a remote HTTP endpoint."""

    def __init__(self, nats_client: NATS, js_context: JetStreamContext, endpoint: Optional[str] = None) -> None:
        self._publisher = Publisher(nats_client, js_context)
        self._subscriber = Subscriber(nats_client, js_context)
        self._endpoint = endpoint or os.getenv("LLM_ENDPOINT", "http://localhost:8000/generate")
        logger.info("RemoteLLM initialized with endpoint %s", self._endpoint)

    async def _generate(self, prompt: str) -> str:
        async with aiohttp.ClientSession() as session:
            async with session.post(self._endpoint, json={"text": prompt}) as resp:
                resp.raise_for_status()
                data = await resp.json()
                text = data.get("text")
                if not isinstance(text, str):
                    raise ValueError("Invalid generate response")
                return text

    async def _handle_memory_event(self, msg: Msg) -> None:
        input_id = "unknown"
        try:
            data = json.loads(msg.data.decode())
            if not isinstance(data, dict):
                raise ValueError("MemoryRetrieved payload must be a dict")
            input_id = data.get("input_id")
            retrieved = data.get("retrieved_knowledge", {})
            facts = retrieved.get("facts", [])
            prompt = ", ".join(map(str, facts))
            logger.info("RemoteLLM generating for %s", input_id)
            response = await self._generate(prompt)
            payload = ResponseGeneratedPayload(
                final_response=response,
                input_id=input_id,
                timestamp=None,
                confidence=0.9,
            )
            await self._publisher.publish(
                EventSubjects.RESPONSE_GENERATED,
                payload,
                use_jetstream=True,
                timeout=10.0,
            )
            await msg.ack()
        except Exception as exc:  # pragma: no cover - runtime network or parse errors
            logger.error("RemoteLLM failed: %s", exc, exc_info=True)
            if hasattr(msg, "nak") and callable(msg.nak):
                try:
                    await msg.nak()
                except Exception:
                    logger.error("Failed to NAK message", exc_info=True)

    async def start_listening(self, durable_name: str = "remote_llm_listener") -> bool:
        if not self._subscriber:
            logger.error("Subscriber not initialized for RemoteLLM.")
            return False
        try:
            await self._subscriber.subscribe(
                subject=EventSubjects.MEMORY_RETRIEVED,
                handler=self._handle_memory_event,
                use_jetstream=True,
                durable=durable_name,
            )
            logger.info("RemoteLLM subscribed to %s", EventSubjects.MEMORY_RETRIEVED)
            return True
        except nats.errors.Error as e:
            logger.error("RemoteLLM failed to subscribe: %s", e, exc_info=True)
            return False
        except Exception as e:
            logger.error("RemoteLLM failed to subscribe: %s", e, exc_info=True)
            return False

    async def stop_listening(self) -> None:
        if self._subscriber:
            await self._subscriber.unsubscribe_all()
            logger.info("RemoteLLM stopped listening.")
