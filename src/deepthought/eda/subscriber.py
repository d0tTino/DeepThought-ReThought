# File: src/deepthought/eda/subscriber.py
import asyncio
import logging
from typing import Awaitable, Callable, Optional

import nats
from nats.aio.client import Client as NATS
from nats.aio.msg import Msg
from nats.js.client import JetStreamContext

logger = logging.getLogger(__name__)
MessageHandlerType = Callable[[Msg], Awaitable[None]]


class Subscriber:
    """A subscriber using a shared NATS client and JetStream context."""

    def __init__(self, nats_client: NATS, js_context: Optional[JetStreamContext] = None):
        """Initialize Subscriber with existing client and optional context."""
        if not nats_client or not nats_client.is_connected:
            raise ValueError("NATS client must be connected.")
        self._nc = nats_client
        self._js = js_context  # Store JS context if provided
        self._subscriptions = []
        logger.debug("Subscriber initialized with shared client.")

    async def subscribe(
        self,
        subject: str,
        handler: MessageHandlerType,
        queue: str = "",
        use_jetstream: bool = False,  # Flag to control behavior
        durable: str = "",
    ) -> None:
        """Subscribe using basic NATS or JetStream."""
        try:
            if use_jetstream:
                if not self._js:
                    raise ValueError("JetStream context required for JetStream subscriptions.")
                if not durable:
                    raise ValueError("Durable name required for JetStream push subscriptions via js.subscribe.")

                # This call binds the handler to the existing durable consumer config on the server
                # Assumes the consumer was created/updated beforehand (e.g., in the test fixture)
                sub = await self._js.subscribe(
                    subject=subject,  # Subject filtering is primarily done by consumer config
                    queue=queue,  # Queue group name (optional for durable)
                    durable=durable,  # Name of the durable consumer config
                    cb=handler,  # Async callback function
                    manual_ack=True,  # IMPORTANT: We must manually ack messages
                )
                logger.info(f"JetStream subscription bound for subject '{subject}' to durable '{durable}'")
            else:
                # Basic NATS subscription
                sub = await self._nc.subscribe(subject=subject, queue=queue, cb=handler)
                logger.info(f"Basic NATS subscription created for '{subject}'")

            self._subscriptions.append(sub)  # Store subscription object for cleanup

        except nats.errors.Error as e:
            logger.error(f"Failed to subscribe to '{subject}' (JetStream={use_jetstream}): {e}", exc_info=True)
            raise e
        except Exception as e:
            logger.error(f"Failed to subscribe to '{subject}' (JetStream={use_jetstream}): {e}", exc_info=True)
            raise e

    async def unsubscribe_all(self) -> None:
        """Unsubscribe from all active subscriptions."""
        if not self._subscriptions:
            return
        logger.info(f"Unsubscribing from {len(self._subscriptions)} subscriptions...")
        tasks = [sub.unsubscribe() for sub in self._subscriptions]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        successful_unsubs = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Error unsubscribing {i}: {result}")
            else:
                successful_unsubs += 1
        logger.info(f"Successfully unsubscribed from {successful_unsubs} subscriptions.")
        self._subscriptions = []

    async def default_handler(self, msg: Msg) -> None:
        """Default handler (should generally not be used if handler is mandatory)."""
        logger.warning(f"Default handler called for message on {msg.subject}.")
        # Ack JetStream messages even in default handler to prevent redelivery
        if hasattr(msg, "ack") and callable(msg.ack):
            try:
                await msg.ack()
                logger.debug(f"Acked message in default handler for subject {msg.subject}")
            except nats.errors.Error as e:
                logger.error(f"Error acking message in default handler: {e}")
        # Basic NATS messages don't need ack.
