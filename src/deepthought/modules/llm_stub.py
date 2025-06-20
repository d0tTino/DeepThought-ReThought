# File: src/deepthought/modules/llm_stub.py
import asyncio
import json
import logging
from datetime import datetime, timezone

import nats
from nats.aio.client import Client as NATS
from nats.aio.msg import Msg
from nats.js.client import JetStreamContext

# Assuming eda modules are in parent dir relative to modules dir
from ..eda.events import EventSubjects, ResponseGeneratedPayload
from ..eda.publisher import Publisher
from ..eda.subscriber import Subscriber

logger = logging.getLogger(__name__)


class LLMStub:
    """Subscribes to MemoryRetrieved, publishes ResponseGenerated via JetStream."""

    def __init__(self, nats_client: NATS, js_context: JetStreamContext):
        """Initialize with shared NATS client and JetStream context."""
        self._publisher = Publisher(nats_client, js_context)
        self._subscriber = Subscriber(nats_client, js_context)
        logger.info("LLMStub initialized (JetStream enabled).")

    async def _handle_memory_event(self, msg: Msg) -> None:
        """Handles MemoryRetrieved event from JetStream."""
        input_id = "unknown"
        data = None
        try:
            raw = json.loads(msg.data.decode())
            if not isinstance(raw, dict):
                logger.warning("Unexpected MemoryRetrieved payload format: %s", type(raw))
                data = {}
            else:
                data = raw
            input_id = data.get("input_id", "unknown")
            retrieved = data.get("retrieved_knowledge", {})
            if isinstance(retrieved, dict) and "retrieved_knowledge" in retrieved:
                knowledge = retrieved.get("retrieved_knowledge", {})
            elif isinstance(retrieved, dict):
                knowledge = retrieved
            else:
                logger.error("retrieved_knowledge is not a dict for input_id %s", input_id)
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
                return

            facts = knowledge.get("facts")
            if not isinstance(facts, list):
                logger.error("retrieved_knowledge missing facts list for input_id %s", input_id)
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
                return

            logger.info(f"LLMStub received memory event ID {input_id}")

            await asyncio.sleep(0.5)  # Simulate work

            facts_str = ", ".join(map(str, facts))
            # Use timezone-aware UTC timestamps
            response = f"Based on: {facts_str}, this is a stub response. [TS: {datetime.now(timezone.utc).isoformat()}]"
            payload = ResponseGeneratedPayload(
                final_response=response,
                input_id=input_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                confidence=0.95,
            )

            logger.info(f"LLMStub: Publishing RESPONSE_GENERATED for input_id: {input_id}")
            try:
                await self._publisher.publish(
                    EventSubjects.RESPONSE_GENERATED,
                    payload,
                    use_jetstream=True,
                    timeout=10.0,
                )
                logger.debug(f"LLMStub: Successfully published RESPONSE_GENERATED for {input_id}")
                await msg.ack()
                logger.debug(f"LLMStub: Acked message for {input_id} in LLMStub")
            except nats.errors.TimeoutError as e:
                logger.error(
                    f"LLMStub: Timeout publishing RESPONSE_GENERATED for {input_id}: {e}",
                    exc_info=True,
                )
                # Do not ack/nak on failure; leave to message broker
            except Exception as e:
                logger.error(
                    f"LLMStub: Failed to publish RESPONSE_GENERATED for {input_id}: {e}",
                    exc_info=True,
                )
                # Do not ack/nak on failure; leave to message broker

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in LLMStub handler: {e}", exc_info=True)
            if hasattr(msg, "ack") and callable(msg.ack):
                try:
                    await msg.ack()
                except nats.errors.Error:
                    logger.error("Failed to ack message after error", exc_info=True)
        except Exception as e:
            logger.error(f"Error in LLMStub handler: {e}", exc_info=True)
            # Do not ack/nak on unexpected errors

    async def start_listening(self, durable_name: str = "llm_stub_listener") -> bool:
        """
        Starts the NATS subscriber to listen for MEMORY_RETRIEVED events.

        Args:
            durable_name: Optional name for the durable consumer. Defaults to "llm_stub_listener".

        Returns:
            bool: True if subscription was successful, False otherwise.
        """
        if not self._subscriber:
            logger.error("Subscriber not initialized for LLMStub.")
            return False

        try:
            logger.info(f"LLMStub subscribing to {EventSubjects.MEMORY_RETRIEVED}...")
            await self._subscriber.subscribe(
                subject=EventSubjects.MEMORY_RETRIEVED,
                handler=self._handle_memory_event,
                use_jetstream=True,
                durable=durable_name,
            )
            logger.info(f"LLMStub successfully subscribed to {EventSubjects.MEMORY_RETRIEVED}.")
            return True
        except nats.errors.Error as e:
            logger.error(f"LLMStub failed to subscribe: {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"LLMStub failed to subscribe: {e}", exc_info=True)
            return False

    async def stop_listening(self) -> None:
        """
        Stops all active NATS subscriptions for this LLMStub instance.
        """
        if self._subscriber:
            logger.info("Stopping LLMStub subscriptions...")
            await self._subscriber.unsubscribe_all()
            logger.info("LLMStub stopped listening.")
        else:
            logger.warning("Cannot stop listening - no subscriber available.")
