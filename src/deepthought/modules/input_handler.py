# File: src/deepthought/modules/input_handler.py
import logging
import uuid
from datetime import datetime, timezone

from nats.aio.client import Client as NATS
from nats.js.client import JetStreamContext

# Assuming eda modules are in parent dir relative to modules dir
from ..eda.events import EventSubjects, InputReceivedPayload
from ..eda.publisher import Publisher

logger = logging.getLogger(__name__)


class InputHandler:
    """Handles user input and publishes InputReceived event via JetStream."""

    def __init__(self, nats_client: NATS, js_context: JetStreamContext):
        """Initialize with shared NATS client and JetStream context."""
        self._publisher = Publisher(nats_client, js_context)
        logger.info("InputHandler initialized (JetStream enabled).")

    async def process_input(self, user_input: str) -> str:
        """Process input and publish via JetStream."""
        if not isinstance(user_input, str):
            raise ValueError("user_input must be a string")

        input_id = str(uuid.uuid4())
        # Use timezone-aware UTC timestamp
        timestamp = datetime.now(timezone.utc).isoformat()
        payload = InputReceivedPayload(user_input=user_input, input_id=input_id, timestamp=timestamp)
        try:
            # Always use JetStream for input events in this version
            await self._publisher.publish(
                EventSubjects.INPUT_RECEIVED, payload, use_jetstream=True, timeout=10.0  # Use JS, increased timeout
            )
            logger.info(f"Published input ID {input_id} (JetStream)")
            return input_id
        except Exception as e:
            logger.error(f"Failed to publish input: {e}", exc_info=True)
            raise
