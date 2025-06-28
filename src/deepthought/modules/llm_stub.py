# File: src/deepthought/modules/llm_stub.py
import asyncio
import json
import logging
from collections import deque
from datetime import datetime, timezone
from typing import Deque, Optional

import nats
from nats.aio.client import Client as NATS
from nats.aio.msg import Msg
from nats.js.client import JetStreamContext

# Assuming eda modules are in parent dir relative to modules dir
from ..config import get_settings
from ..eda.events import EventSubjects, ResponseGeneratedPayload
from ..eda.publisher import Publisher
from ..eda.subscriber import Subscriber

logger = logging.getLogger(__name__)


class LLMStub:
    """Subscribes to MemoryRetrieved, publishes ResponseGenerated via JetStream."""

    def __init__(
        self,
        nats_client: NATS,
        js_context: JetStreamContext,
        reward_buffer_size: Optional[int] = None,
    ):
        """Initialize with shared NATS client and JetStream context."""
        self._publisher = Publisher(nats_client, js_context)
        self._subscriber = Subscriber(nats_client, js_context)
        buffer_size = reward_buffer_size or get_settings().reward.buffer_size
        self._recent_rewards: Deque[float] = deque(maxlen=buffer_size)
        logger.info("LLMStub initialized (JetStream enabled).")

    async def _handle_memory_event(self, msg: Msg) -> None:
        """Handles MemoryRetrieved event from JetStream."""
        input_id = "unknown"
        data = None
        try:
            data = json.loads(msg.data.decode())
            if not isinstance(data, dict):
                raise ValueError(
                    f"Unexpected MemoryRetrieved payload format: {type(data)}"
                )
            input_id = data.get("input_id")
            retrieved = data.get("retrieved_knowledge")
            if not isinstance(input_id, str) or retrieved is None:
                raise ValueError("Invalid memory payload fields")
            if isinstance(retrieved, dict) and "retrieved_knowledge" in retrieved:
                knowledge = retrieved.get("retrieved_knowledge", {})
            elif isinstance(retrieved, dict):
                knowledge = retrieved
            else:
                logger.error(
                    "retrieved_knowledge is not a dict for input_id %s", input_id
                )
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
                logger.error(
                    "retrieved_knowledge missing facts list for input_id %s", input_id
                )
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

            logger.info(
                f"LLMStub: Publishing RESPONSE_GENERATED for input_id: {input_id}"
            )
            try:
                await self._publisher.publish(
                    EventSubjects.RESPONSE_GENERATED,
                    payload,
                    use_jetstream=True,
                    timeout=10.0,
                )
                logger.debug(
                    f"LLMStub: Successfully published RESPONSE_GENERATED for {input_id}"
                )
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

        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Invalid MemoryRetrieved payload: {e}", exc_info=True)
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
            logger.error(f"Error in LLMStub handler: {e}", exc_info=True)
            # Do not ack/nak on unexpected errors

    async def _handle_reward_event(self, msg: Msg) -> None:
        """Store rewards published on ``agent.reward``."""
        try:
            data = json.loads(msg.data.decode())
            if not isinstance(data, dict) or "reward" not in data:
                raise ValueError("payload must contain reward field")
            reward = float(data["reward"])
            self._recent_rewards.append(reward)
        except Exception as exc:  # pragma: no cover - invalid payload
            logger.error("Invalid agent.reward payload: %s", exc)
        finally:
            if hasattr(msg, "ack") and callable(msg.ack):
                try:
                    await msg.ack()
                except Exception:  # pragma: no cover - ack issues
                    logger.error("Failed to ack reward message", exc_info=True)

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
            await self._subscriber.subscribe(
                subject="agent.reward",
                handler=self._handle_reward_event,
                use_jetstream=True,
                durable=f"{durable_name}_reward",
            )
            logger.info(
                f"LLMStub successfully subscribed to {EventSubjects.MEMORY_RETRIEVED}."
            )
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
