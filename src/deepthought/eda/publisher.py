# File: src/deepthought/eda/publisher.py
import logging
from typing import Any, Dict, Optional, Union

import nats
from nats.aio.client import Client as NATS
from nats.js.client import JetStreamContext

logger = logging.getLogger(__name__)


class Publisher:
    """A publisher using a shared NATS client and JetStream context."""

    def __init__(self, nats_client: NATS, js_context: JetStreamContext):
        """Initialize Publisher with existing client and context."""
        if not nats_client or not nats_client.is_connected:
            raise ValueError("NATS client must be connected.")
        if not js_context:
            raise ValueError("JetStream context must be provided.")
        self._nc = nats_client
        self._js = js_context
        logger.debug("Publisher initialized with shared client and JS context.")

    async def publish(
        self,
        subject: str,
        payload: Union[str, Dict, Any],
        use_jetstream: bool = True,
        timeout: float = 10.0,
    ) -> Optional[Dict]:
        """Publish message, using JetStream if requested."""
        # Convert payload
        if isinstance(payload, bytes):
            data = payload
        elif isinstance(payload, str):
            data = payload.encode()
        elif hasattr(payload, "to_json"):
            data = payload.to_json().encode()
        elif isinstance(payload, (dict, list)):
            import json

            data = json.dumps(payload).encode()
        else:
            data = str(payload).encode()

        payload_summary = str(payload)
        if len(payload_summary) > 100:
            payload_summary = payload_summary[:97] + "..."

        try:
            if use_jetstream:
                # Use JetStream publish with timeout
                ack = await self._js.publish(subject, data, timeout=timeout)  # Use timeout
                logger.debug(f"Published to '{subject}' via JetStream: seq={ack.seq}")
                return {"seq": ack.seq, "stream": ack.stream}
            else:
                # Use regular NATS publish
                await self._nc.publish(subject, data)
                logger.debug(f"Published basic NATS message to '{subject}'")
                return None
        except nats.errors.TimeoutError as e:
            message = f"Publish timeout for '{subject}' with payload {payload_summary}: {e}"
            logger.error(message, exc_info=True)
            raise type(e)(message) from e
        except Exception as e:
            message = f"Failed to publish to '{subject}' with payload {payload_summary}: {e}"
            logger.error(message, exc_info=True)  # Log traceback
            raise type(e)(message) from e
