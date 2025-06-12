import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from nats.aio.client import Client as NATS
from nats.aio.msg import Msg
from nats.js.client import JetStreamContext

from ..config import settings
from ..eda.events import EventSubjects, MemoryRetrievedPayload
from ..eda.publisher import Publisher
from ..eda.subscriber import Subscriber

logger = logging.getLogger(__name__)


class BasicMemory:
    """File-backed memory module storing past inputs."""

    def __init__(
        self,
        nats_client: NATS,
        js_context: JetStreamContext,
        memory_file: Optional[str] = None,
    ) -> None:
        self._publisher = Publisher(nats_client, js_context)
        self._subscriber = Subscriber(nats_client, js_context)
        self._memory_file = memory_file or settings.memory_file

        if not os.path.exists(self._memory_file):
            with open(self._memory_file, "w", encoding="utf-8") as f:
                json.dump([], f)
        logger.info("BasicMemory initialized with file %s", self._memory_file)

    def _read_memory(self) -> List[Dict[str, Any]]:
        try:
            with open(self._memory_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def _write_memory(self, data: List[Dict[str, Any]]) -> None:
        with open(self._memory_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

    async def _handle_input_event(self, msg: Msg) -> None:
        input_id = "unknown"
        try:
            data = json.loads(msg.data.decode())
            input_id = data.get("input_id", "unknown")
            user_input = data.get("user_input", "")
            logger.info("BasicMemory received input event ID %s", input_id)

            history = self._read_memory()
            history.append({"timestamp": datetime.now(timezone.utc).isoformat(), "user_input": user_input})
            self._write_memory(history)

            last_entries = history[-3:]
            facts = [entry["user_input"] for entry in last_entries]
            memory_data = {"facts": facts, "source": "basic_memory"}
            payload = MemoryRetrievedPayload(
                retrieved_knowledge={"retrieved_knowledge": memory_data},
                input_id=input_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

            await self._publisher.publish(EventSubjects.MEMORY_RETRIEVED, payload, use_jetstream=True, timeout=10.0)
            logger.info("BasicMemory published memory event ID %s", input_id)
            await msg.ack()
        except Exception as e:
            logger.error("Error in BasicMemory handler: %s", e, exc_info=True)
            if hasattr(msg, "nak") and callable(msg.nak):
                try:
                    await msg.nak()
                except Exception:
                    logger.error("Failed to NAK message", exc_info=True)
            elif hasattr(msg, "ack") and callable(msg.ack):
                try:
                    await msg.ack()
                except Exception:
                    logger.error("Failed to ack message after error", exc_info=True)

    async def start_listening(self, durable_name: str = "memory_basic_listener") -> bool:
        if not self._subscriber:
            logger.error("Subscriber not initialized for BasicMemory.")
            return False
        try:
            await self._subscriber.subscribe(
                subject=EventSubjects.INPUT_RECEIVED,
                handler=self._handle_input_event,
                use_jetstream=True,
                durable=durable_name,
            )
            logger.info("BasicMemory subscribed to %s", EventSubjects.INPUT_RECEIVED)
            return True
        except Exception as e:
            logger.error("BasicMemory failed to subscribe: %s", e, exc_info=True)
            return False

    async def stop_listening(self) -> None:
        if self._subscriber:
            await self._subscriber.unsubscribe_all()
            logger.info("BasicMemory stopped listening.")
        else:
            logger.warning("Cannot stop listening - no subscriber available.")
