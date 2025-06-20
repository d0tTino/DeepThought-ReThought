# File: src/deepthought/modules/memory_stub.py
import asyncio
import json
import logging
from datetime import datetime, timezone

import nats
from nats.aio.client import Client as NATS
from nats.aio.msg import Msg
from nats.js.client import JetStreamContext

# Assuming eda modules are in parent dir relative to modules dir
from ..eda.events import EventSubjects, MemoryRetrievedPayload
from ..eda.publisher import Publisher
from ..eda.subscriber import Subscriber

logger = logging.getLogger(__name__)


class MemoryStub:
    """Subscribes to InputReceived, publishes MemoryRetrieved via JetStream."""

    def __init__(self, nats_client: NATS, js_context: JetStreamContext):
        """Initialize with shared NATS client and JetStream context."""
        self._publisher = Publisher(nats_client, js_context)
        self._subscriber = Subscriber(nats_client, js_context)
        logger.info("MemoryStub initialized (JetStream enabled).")

    async def _handle_input_event(self, msg: Msg) -> None:
        """Handles InputReceived event from JetStream."""
        input_id = "unknown"
        data = None
        try:
            data = json.loads(msg.data.decode())
            if not isinstance(data, dict):
                raise ValueError("InputReceived payload must be a dict")
            input_id = data.get("input_id")
            user_input = data.get("user_input")
            if not isinstance(input_id, str) or not isinstance(user_input, str):
                raise ValueError("Invalid input payload fields")
            logger.info(f"MemoryStub received input event ID {input_id}")

            await asyncio.sleep(0.1)  # Simulate work

            # Align with LLMStub expectation which looks for a nested
            # "retrieved_knowledge" block within the event payload.
            memory_data = {
                "facts": ["Fact1", f"User asked: {user_input}"],
                "source": "memory_stub",
            }
            payload = MemoryRetrievedPayload(
                retrieved_knowledge={"retrieved_knowledge": memory_data},
                input_id=input_id,
                # Use timezone-aware UTC timestamp
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

            # Publish result via JetStream
            await self._publisher.publish(
                EventSubjects.MEMORY_RETRIEVED,
                payload,
                use_jetstream=True,
                timeout=10.0,
            )
            logger.info(f"MemoryStub published memory event ID {input_id}")

            # Acknowledge the received message
            await msg.ack()
            logger.debug(f"Acked message for {input_id} in MemoryStub")

        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Invalid InputReceived payload: {e}", exc_info=True)
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
            logger.error(f"Error in MemoryStub handler: {e}", exc_info=True)
            if hasattr(msg, "nak") and callable(msg.nak):
                try:
                    await msg.nak()
                except nats.errors.Error:
                    logger.error("Failed to NAK message", exc_info=True)
            elif hasattr(msg, "ack") and callable(msg.ack):
                try:
                    await msg.ack()
                except nats.errors.Error:
                    logger.error("Failed to ack message after error", exc_info=True)

    async def start_listening(self, durable_name: str = "memory_stub_listener") -> bool:
        """
        Starts the NATS subscriber to listen for INPUT_RECEIVED events.

        Args:
            durable_name: Optional name for the durable consumer. Defaults to "memory_stub_listener".

        Returns:
            bool: True if subscription was successful, False otherwise.
        """
        if not self._subscriber:
            logger.error("Subscriber not initialized for MemoryStub.")
            return False

        try:
            logger.info(f"MemoryStub subscribing to {EventSubjects.INPUT_RECEIVED}...")
            await self._subscriber.subscribe(
                subject=EventSubjects.INPUT_RECEIVED,
                handler=self._handle_input_event,
                use_jetstream=True,
                durable=durable_name,
            )
            logger.info(f"MemoryStub successfully subscribed to {EventSubjects.INPUT_RECEIVED}.")
            return True
        except nats.errors.Error as e:
            logger.error(f"MemoryStub failed to subscribe: {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"MemoryStub failed to subscribe: {e}", exc_info=True)
            return False

    async def stop_listening(self) -> None:
        """
        Stops all active NATS subscriptions for this MemoryStub instance.
        """
        if self._subscriber:
            logger.info("Stopping MemoryStub subscriptions...")
            await self._subscriber.unsubscribe_all()
            logger.info("MemoryStub stopped listening.")
        else:
            logger.warning("Cannot stop listening - no subscriber available.")
