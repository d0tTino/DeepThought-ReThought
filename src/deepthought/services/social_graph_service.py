import json
import logging
from datetime import datetime, timezone
from typing import Optional

from nats.aio.client import Client as NATS
from nats.aio.msg import Msg
from nats.js.client import JetStreamContext

from ..eda.events import EventSubjects, MemoryRetrievedPayload
from ..eda.publisher import Publisher
from ..eda.subscriber import Subscriber
from .db_manager import DBManager

logger = logging.getLogger(__name__)


class SocialGraphService:
    """Service that stores interactions using :class:`DBManager`."""

    def __init__(
        self,
        nats_client: NATS,
        js_context: JetStreamContext,
        db: Optional[DBManager] = None,
    ) -> None:
        self._publisher = Publisher(nats_client, js_context)
        self._subscriber = Subscriber(nats_client, js_context)
        self._db = db or DBManager()

    async def _handle_input(self, msg: Msg) -> None:
        input_id = "unknown"
        try:
            data = json.loads(msg.data.decode())
            if not isinstance(data, dict):
                raise ValueError("InputReceived payload must be a dict")
            input_id = data.get("input_id")
            user_input = data.get("user_input")
            if not isinstance(input_id, str) or not isinstance(user_input, str):
                raise ValueError("Invalid input payload fields")
            logger.info("SocialGraphService received input event ID %s", input_id)

            await self._db.store_memory("user", user_input)
            await self._db.log_interaction("user", None)
            rows = await self._db.recall_user("user")
            facts = [m[1] for m in rows][-3:]
            payload = MemoryRetrievedPayload(
                retrieved_knowledge={"facts": facts, "source": "social_graph_service"},
                input_id=input_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

            await self._publisher.publish(
                EventSubjects.MEMORY_RETRIEVED,
                payload,
                use_jetstream=True,
                timeout=10.0,
            )
            logger.info("SocialGraphService published memory event ID %s", input_id)
            await msg.ack()
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("Invalid InputReceived payload: %s", e, exc_info=True)
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
        except Exception as e:
            logger.error("Error in SocialGraphService handler: %s", e, exc_info=True)
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

    async def start(self, durable_name: str = "social_graph_listener") -> bool:
        if self._subscriber is None:
            logger.error("Subscriber not initialized for SocialGraphService.")
            return False
        try:
            await self._subscriber.subscribe(
                subject=EventSubjects.INPUT_RECEIVED,
                handler=self._handle_input,
                use_jetstream=True,
                durable=durable_name,
            )
            logger.info("SocialGraphService subscribed to %s", EventSubjects.INPUT_RECEIVED)
            return True
        except Exception as e:
            logger.error("SocialGraphService failed to subscribe: %s", e, exc_info=True)
            return False

    async def stop(self) -> None:
        if self._subscriber:
            await self._subscriber.unsubscribe_all()
            logger.info("SocialGraphService stopped listening.")
        else:
            logger.warning("Cannot stop listening - no subscriber available.")

    async def __aenter__(self) -> "SocialGraphService":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.stop()
