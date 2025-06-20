# File: src/deepthought/modules/output_handler.py
import json
import logging
from collections import OrderedDict
from typing import Callable, Dict, Optional

import nats
from nats.aio.client import Client as NATS
from nats.aio.msg import Msg
from nats.js.client import JetStreamContext

# Assuming eda modules are in parent dir relative to modules dir
from ..eda.events import EventSubjects
from ..eda.subscriber import Subscriber

logger = logging.getLogger(__name__)


class OutputHandler:
    """Subscribes to ResponseGenerated via JetStream and handles output."""

    def __init__(
        self,
        nats_client: NATS,
        js_context: JetStreamContext,
        output_callback: Optional[Callable[[str, str], None]] = None,
        max_responses: int = 100,
    ):
        """Initialize with shared NATS client and JetStream context."""
        self._subscriber = Subscriber(nats_client, js_context)
        self._responses: "OrderedDict[str, str]" = OrderedDict()
        self._output_callback = output_callback
        self._max_responses = max_responses
        logger.info("OutputHandler initialized (JetStream enabled).")

    async def _handle_response_event(self, msg: Msg) -> None:
        """Handles ResponseGenerated event from JetStream."""
        input_id = "unknown"
        data = None
        try:
            data = json.loads(msg.data.decode())
            if not isinstance(data, dict):
                raise ValueError("ResponseGenerated payload must be a dict")
            input_id = data.get("input_id")
            final_response = data.get("final_response")
            if not isinstance(input_id, str) or not isinstance(final_response, str):
                raise ValueError("Invalid response payload fields")
            logger.info(f"OutputHandler received response event ID {input_id}")

            self._responses[input_id] = final_response  # Store response
            if len(self._responses) > self._max_responses:
                self._responses.popitem(last=False)

            # Use callback or log when no callback provided
            if self._output_callback:
                self._output_callback(input_id, final_response)
            else:
                logger.info(f"Output ({input_id}): {final_response}")

            # Acknowledge the received message
            await msg.ack()
            logger.debug(f"Acked message for {input_id} in OutputHandler")

        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Invalid ResponseGenerated payload: {e}", exc_info=True)
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
            logger.error(f"Error in OutputHandler handler: {e}", exc_info=True)
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

    async def start_listening(self, durable_name: str = "output_handler_listener") -> bool:
        """
        Starts the NATS subscriber to listen for RESPONSE_GENERATED events.

        Args:
            durable_name: Optional name for the durable consumer. Defaults to "output_handler_listener".

        Returns:
            bool: True if subscription was successful, False otherwise.
        """
        if not self._subscriber:
            logger.error("Subscriber not initialized for OutputHandler.")
            return False

        try:
            logger.info(f"OutputHandler subscribing to {EventSubjects.RESPONSE_GENERATED}...")
            await self._subscriber.subscribe(
                subject=EventSubjects.RESPONSE_GENERATED,
                handler=self._handle_response_event,
                use_jetstream=True,
                durable=durable_name,
            )
            logger.info(f"OutputHandler successfully subscribed to {EventSubjects.RESPONSE_GENERATED}.")
            return True
        except nats.errors.Error as e:
            logger.error(f"OutputHandler failed to subscribe: {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"OutputHandler failed to subscribe: {e}", exc_info=True)
            return False

    async def stop_listening(self) -> None:
        """
        Stops all active NATS subscriptions for this OutputHandler instance.
        """
        if self._subscriber:
            logger.info("Stopping OutputHandler subscriptions...")
            await self._subscriber.unsubscribe_all()
            logger.info("OutputHandler stopped listening.")
        else:
            logger.warning("Cannot stop listening - no subscriber available.")

    # Methods to retrieve responses for testing
    def get_response(self, input_id: str) -> Optional[str]:
        return self._responses.get(input_id)

    def get_all_responses(self) -> Dict[str, str]:
        return self._responses
