import logging
import uuid
from datetime import datetime, timezone

import nats
from nats.aio.client import Client as NATS
from nats.js.client import JetStreamContext

from ..eda.events import EventSubjects, InputReceivedPayload, MemoryRetrievedPayload
from ..eda.publisher import Publisher

logger = logging.getLogger(__name__)


class InputHandler:
    """Handles user input and publishes InputReceived events."""

    def __init__(self, nats_client: NATS, js_context: JetStreamContext, cognitive_service=None):
        """Initialize with optional cognitive memory service."""
        self._publisher = Publisher(nats_client, js_context)
        self._memory_service = cognitive_service
        logger.info("InputHandler initialized (JetStream enabled).")

    async def process_input(
        self,
        user_input: str,
        *,
        message_id: str | None = None,
        channel_id: str | None = None,
        guild_id: str | None = None,
        author_id: str | None = None,
        author_name: str | None = None,
        author_is_bot: bool | None = None,
    ) -> str:
        """Process input and publish via JetStream."""
        if not isinstance(user_input, str):
            raise ValueError("user_input must be a string")

        input_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        payload = InputReceivedPayload(
            user_input=user_input,
            input_id=input_id,
            timestamp=timestamp,
            message_id=message_id,
            channel_id=channel_id,
            guild_id=guild_id,
            author_id=author_id,
            author_name=author_name,
            author_is_bot=author_is_bot,
        )
        try:
            await self._publisher.publish(
                EventSubjects.INPUT_RECEIVED,
                payload,
                use_jetstream=True,
                timeout=10.0,
            )
            await self._publisher.publish(
                EventSubjects.CHAT_RAW,
                user_input,
                use_jetstream=True,
                timeout=10.0,
            )
            logger.info("Published input ID %s (JetStream)", input_id)

            if self._memory_service is not None:
                context = []
                try:
                    context = self._memory_service.retrieve_context(user_input)
                except Exception as err:  # pragma: no cover - defensive
                    logger.error("Memory retrieval failed: %s", err, exc_info=True)

                mem_payload = MemoryRetrievedPayload(
                    retrieved_knowledge={"facts": context, "source": "hierarchical"},
                    user_input=user_input,
                    input_id=input_id,
                    channel_id=channel_id,
                    author_id=author_id,
                    author_name=author_name,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )

                await self._publisher.publish(
                    EventSubjects.MEMORY_RETRIEVED,
                    mem_payload,
                    use_jetstream=True,
                    timeout=10.0,
                )
                logger.info("Published memory for input ID %s", input_id)
            return input_id
        except nats.errors.TimeoutError as e:
            logger.error(f"NATS timeout publishing input: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Failed to publish input: {e}", exc_info=True)
            raise
