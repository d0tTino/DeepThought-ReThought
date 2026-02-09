from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Protocol

from nats.aio.client import Client as NATS
from nats.aio.msg import Msg
from nats.js.client import JetStreamContext

from ..eda.events import EventSubjects, InputReceivedPayload, ResponseRankedPayload
from .base import BaseService

logger = logging.getLogger(__name__)


class _Channel(Protocol):
    async def send(self, content: str) -> None: ...


class _DiscordClient(Protocol):
    def get_channel(self, channel_id: int) -> _Channel | None: ...


class DiscordGatewayService(BaseService):
    """Bridge Discord messages to and from the DeepThought event bus."""

    def __init__(
        self,
        nats_client: NATS | None = None,
        js_context: JetStreamContext | None = None,
        *,
        discord_client: _DiscordClient | None = None,
        response_subject: str = EventSubjects.RESPONSE_RANKED,
        durable_name: str = "discord_gateway_response_ranked",
        nats_url: str | None = None,
        connect_retries: int = 1,
        connect_timeout: float = 2.0,
    ) -> None:
        super().__init__(
            nats_client,
            js_context,
            nats_url=nats_url,
            connect_retries=connect_retries,
            connect_timeout=connect_timeout,
        )
        self._discord_client = discord_client
        self._pending_channels: dict[str, str] = {}
        self.add_subscription(
            response_subject,
            self._handle_ranked_response,
            durable=durable_name,
            use_jetstream=True,
        )

    @staticmethod
    def build_input_payload(message: Any, text: str | None = None) -> InputReceivedPayload:
        """Create a reusable INPUT_RECEIVED payload from a Discord message-like object."""

        user_text = text if text is not None else str(getattr(message, "content", ""))
        author = getattr(message, "author", None)
        channel = getattr(message, "channel", None)
        guild = getattr(message, "guild", None)
        return InputReceivedPayload(
            user_input=user_text,
            input_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            message_id=(str(getattr(message, "id", "")) if getattr(message, "id", None) is not None else None),
            channel_id=(str(getattr(channel, "id", "")) if getattr(channel, "id", None) is not None else None),
            guild_id=(str(getattr(guild, "id", "")) if getattr(guild, "id", None) is not None else None),
            author_id=(str(getattr(author, "id", "")) if getattr(author, "id", None) is not None else None),
            author_name=(getattr(author, "display_name", None) or getattr(author, "name", None)),
            author_is_bot=bool(getattr(author, "bot", False)) if author is not None else None,
        )

    async def handle_discord_message(self, message: Any) -> str | None:
        """Publish human-authored Discord messages as INPUT_RECEIVED events."""

        author = getattr(message, "author", None)
        if getattr(author, "bot", False):
            return None
        if not self._publisher:
            logger.warning("DiscordGatewayService publisher unavailable; dropping message")
            return None

        payload = self.build_input_payload(message)
        if payload.input_id and payload.channel_id:
            self._pending_channels[payload.input_id] = payload.channel_id

        await self._publisher.publish(
            EventSubjects.INPUT_RECEIVED,
            payload,
            use_jetstream=True,
            timeout=10.0,
        )
        return payload.input_id

    async def _send_channel_message(self, channel_id: str, content: str) -> bool:
        if self._discord_client is None:
            logger.warning("Discord client unavailable; cannot forward ranked response")
            return False
        try:
            channel = self._discord_client.get_channel(int(channel_id))
        except (TypeError, ValueError):
            logger.warning("Invalid channel id: %s", channel_id)
            return True
        if channel is None:
            logger.warning("Channel %s not found", channel_id)
            return True
        await channel.send(content)
        return True

    async def _handle_ranked_response(self, msg: Msg) -> None:
        try:
            data = json.loads(msg.data.decode())
            if not isinstance(data, dict):
                raise ValueError("ResponseRanked payload must be a dict")
            payload = ResponseRankedPayload.from_dict(data)
            if not payload.final_response:
                raise ValueError("final_response is required")

            channel_id = payload.channel_id
            if not channel_id and payload.input_id:
                channel_id = self._pending_channels.get(payload.input_id)
            if not channel_id:
                logger.warning("No channel mapping found for ranked response input_id=%s", payload.input_id)
                await msg.ack()
                return

            sent = await self._send_channel_message(channel_id, payload.final_response)
            if sent:
                if payload.input_id:
                    self._pending_channels.pop(payload.input_id, None)
                await msg.ack()
            elif hasattr(msg, "nak") and callable(msg.nak):
                await msg.nak()
            else:
                await msg.ack()
        except (json.JSONDecodeError, ValueError):
            logger.error("Invalid ranked response payload", exc_info=True)
            if hasattr(msg, "nak") and callable(msg.nak):
                await msg.nak()
            elif hasattr(msg, "ack") and callable(msg.ack):
                await msg.ack()
        except Exception:
            logger.error("Failed to handle ranked response", exc_info=True)
            if hasattr(msg, "nak") and callable(msg.nak):
                await msg.nak()
