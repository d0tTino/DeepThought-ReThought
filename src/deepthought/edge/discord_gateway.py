"""Discord gateway for DeepThought's event bus."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

try:  # pragma: no cover - import guard for optional dependency
    import discord
except ModuleNotFoundError:  # pragma: no cover - fallback for tests without discord.py
    discord = None  # type: ignore

from nats.aio.msg import Msg

from ..eda.events import EventSubjects
from ..eda.publisher import Publisher
from ..eda.subscriber import Subscriber

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DiscordGatewayConfig:
    """Configuration for the Discord gateway."""

    incoming_subject: str = "discord.message.text"
    response_subject: str = EventSubjects.RESPONSE_RANKED
    durable_name: str = "gw_ranked_v1"


class DiscordGateway:
    """Bridge Discord messages to the DeepThought event bus."""

    def __init__(
        self,
        publisher: Publisher,
        subscriber: Subscriber,
        *,
        client: Optional[Any] = None,
        config: Optional[DiscordGatewayConfig] = None,
    ) -> None:
        self._publisher = publisher
        self._subscriber = subscriber
        self._config = config or DiscordGatewayConfig()
        self._client = client or self._create_default_client()
        self._response_subscription_started = False
        self._registered_events = False
        logger.debug("DiscordGateway initialized with subjects %s -> %s", self._config.incoming_subject, self._config.response_subject)

    def _create_default_client(self) -> Any:
        """Create a default Discord client instance."""
        if discord is None:
            raise RuntimeError(
                "discord.py is not installed. Provide a pre-configured client when constructing DiscordGateway."
            )
        intents = discord.Intents.default()
        intents.message_content = True
        return discord.Client(intents=intents)

    def _register_client_events(self) -> None:
        if self._registered_events:
            return

        @self._client.event
        async def on_ready() -> None:  # pragma: no cover - trivial logging
            logger.info("Discord gateway connected as %%s", getattr(self._client, "user", "unknown"))

        @self._client.event
        async def on_message(message: Any) -> None:
            await self._handle_discord_message(message)

        self._registered_events = True
        logger.debug("Discord client events registered")

    async def _handle_discord_message(self, message: Any) -> None:
        """Translate a Discord message into a bus event."""
        author = getattr(message, "author", None)
        if author and getattr(author, "bot", False):
            logger.debug("Ignoring bot message from %s", getattr(author, "id", "unknown"))
            return

        content = getattr(message, "content", None)
        if not content:
            logger.debug("Ignoring Discord message without content")
            return

        payload: Dict[str, Any] = {
            "message_id": getattr(message, "id", None),
            "channel_id": getattr(getattr(message, "channel", None), "id", None),
            "author_id": getattr(author, "id", None),
            "author_name": getattr(author, "name", None),
            "content": content,
        }
        logger.info("Publishing Discord message %s to %s", payload["message_id"], self._config.incoming_subject)
        await self._publisher.publish(self._config.incoming_subject, payload, use_jetstream=True)

    async def _handle_ranked_response(self, msg: Msg) -> None:
        """Post ranked responses back into Discord."""
        try:
            data = json.loads(msg.data.decode()) if msg.data else {}
        except json.JSONDecodeError as exc:
            logger.error("Failed to decode ranked response payload: %s", exc)
            await msg.ack()
            return

        channel_id = data.get("channel_id")
        message_content = self._extract_message_content(data)

        if not channel_id or not message_content:
            logger.warning("Ranked response missing channel or content: %s", data)
            await msg.ack()
            return

        channel = None
        if hasattr(self._client, "get_channel"):
            channel = self._client.get_channel(channel_id)

        if channel is None:
            logger.warning("Discord channel %s not found for response", channel_id)
            await msg.ack()
            return

        send_method: Optional[Callable[[str], Any]] = getattr(channel, "send", None)
        if send_method is None:
            logger.error("Channel %s does not support async send", channel_id)
            await msg.ack()
            return

        logger.info("Sending ranked response to channel %s", channel_id)
        result = send_method(message_content)
        if asyncio.iscoroutine(result):
            await result
        else:
            logger.error("Channel %s send method did not return a coroutine", channel_id)
            await msg.ack()
            return
        await msg.ack()

    @staticmethod
    def _extract_message_content(data: Dict[str, Any]) -> Optional[str]:
        message_content = data.get("content") or data.get("final_response")
        if message_content:
            return message_content
        ranked = data.get("ranked_candidates")
        if isinstance(ranked, list) and ranked:
            first = ranked[0]
            if isinstance(first, dict):
                return first.get("content") or first.get("text")
            if isinstance(first, str):
                return first
        return None

    async def setup(self) -> None:
        """Prepare the gateway by registering events and subscriptions."""
        self._register_client_events()
        if not self._response_subscription_started:
            await self._subscriber.subscribe(
                subject=self._config.response_subject,
                handler=self._handle_ranked_response,
                use_jetstream=True,
                durable=self._config.durable_name,
            )
            self._response_subscription_started = True
            logger.debug(
                "Subscribed to %s with durable %s", self._config.response_subject, self._config.durable_name
            )

    async def run(self, token: str) -> None:
        """Run the Discord client after ensuring subscriptions are ready."""
        await self.setup()
        await self._client.start(token)

    async def stop(self) -> None:
        """Cleanly shut down the gateway."""
        if self._response_subscription_started:
            await self._subscriber.unsubscribe_all()
            self._response_subscription_started = False
        close_method = getattr(self._client, "close", None)
        if close_method is not None:
            result = close_method()
            if asyncio.iscoroutine(result):
                await result
        logger.info("Discord gateway stopped")
