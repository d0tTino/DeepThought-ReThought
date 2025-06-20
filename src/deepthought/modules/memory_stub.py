# File: src/deepthought/modules/memory_stub.py
import asyncio
import json
import logging
from datetime import datetime
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
        try:
            data = json.loads(msg.data.decode())
            input_id = data.get("input_id", "unknown")
            user_input = data.get("user_input", "")
            logger.info(f"MemoryStub received input event ID {input_id}")

            await asyncio.sleep(0.1) # Simulate work

            memory_data = {
                "retrieved_knowledge": {
                    "facts": ["Fact1", f"User asked: {user_input}"],
                    "source": "memory_stub"
                }
            }
            payload = MemoryRetrievedPayload(
                retrieved_knowledge=memory_data,
                input_id=input_id,
                timestamp=datetime.utcnow().isoformat()
            )

            # Publish result via JetStream
            await self._publisher.publish(
                EventSubjects.MEMORY_RETRIEVED, payload,
                use_jetstream=True, timeout=10.0
            )
            logger.info(f"MemoryStub published memory event ID {input_id}")

            # Acknowledge the received message
            await msg.ack()
            logger.debug(f"Acked message for {input_id} in MemoryStub")

        except Exception as e:
            logger.error(f"Error in MemoryStub handler: {e}", exc_info=True)
            # Optionally NAK the message if error is temporary:
            # if hasattr(msg, 'nak') and callable(msg.nak): await msg.nak()

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
                durable=durable_name
            )
            logger.info(f"MemoryStub successfully subscribed to {EventSubjects.INPUT_RECEIVED}.")
            return True
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