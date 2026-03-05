from __future__ import annotations

import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from time import monotonic
from typing import Any, Awaitable, Callable, Protocol

from nats.aio.client import Client as NATS
from nats.aio.msg import Msg
from nats.js.client import JetStreamContext

from ..eda.events import EventSubjects, InputReceivedPayload, ResponseRankedPayload
from .base import BaseService
from .human_interaction_policy import HumanInteractionPolicy

logger = logging.getLogger(__name__)


class _Channel(Protocol):
    async def send(self, content: str, **kwargs: Any) -> None: ...


class _DiscordClient(Protocol):
    def get_channel(self, channel_id: int) -> _Channel | None: ...


@dataclass
class _PendingRoute:
    channel_id: str
    source_message_id: str | None
    thread_id: str | None
    author_id: str | None


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
        interaction_policy: HumanInteractionPolicy | None = None,
        clock: Callable[[], float] = monotonic,
        sleeper: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        super().__init__(
            nats_client,
            js_context,
            nats_url=nats_url,
            connect_retries=connect_retries,
            connect_timeout=connect_timeout,
        )
        self._discord_client = discord_client
        self._pending_routes: dict[str, _PendingRoute] = {}
        self._interaction_policy = interaction_policy or HumanInteractionPolicy()
        self._clock = clock
        self._sleep = sleeper
        self._recent_channel_activity: dict[str, list[float]] = {}
        self._familiarity_counts: dict[tuple[str, str], int] = {}
        self._cooldown_until: dict[str, float] = {}
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
        raw_attachments = getattr(message, "attachments", None)
        reference = getattr(message, "reference", None)
        thread = getattr(message, "thread", None)
        attachments = []
        if isinstance(raw_attachments, (list, tuple)):
            for attachment in raw_attachments:
                descriptor = InputReceivedPayload.AttachmentDescriptor.from_dict(
                    {
                        "url": getattr(attachment, "url", None),
                        "content_type": getattr(attachment, "content_type", None),
                        "filename": getattr(attachment, "filename", None),
                        "size": getattr(attachment, "size", None),
                    }
                )
                if descriptor is not None:
                    attachments.append(descriptor)

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
            reference_message_id=(
                str(getattr(reference, "message_id", "")) if getattr(reference, "message_id", None) is not None else None
            ),
            thread_id=(str(getattr(thread, "id", "")) if getattr(thread, "id", None) is not None else None),
            attachments=attachments or None,
        )

    def _record_inbound_activity(self, *, channel_id: str | None, author_id: str | None) -> None:
        now = self._clock()
        if channel_id:
            window = [ts for ts in self._recent_channel_activity.get(channel_id, []) if now - ts <= 60.0]
            window.append(now)
            self._recent_channel_activity[channel_id] = window
            if author_id:
                key = (channel_id, author_id)
                self._familiarity_counts[key] = self._familiarity_counts.get(key, 0) + 1

    def _estimate_channel_pace(self, channel_id: str | None) -> float:
        if not channel_id:
            return 0.0
        now = self._clock()
        active = [ts for ts in self._recent_channel_activity.get(channel_id, []) if now - ts <= 60.0]
        self._recent_channel_activity[channel_id] = active
        return float(len(active))

    def _estimate_familiarity(self, channel_id: str | None, author_id: str | None) -> float:
        if not channel_id or not author_id:
            return 0.0
        count = self._familiarity_counts.get((channel_id, author_id), 0)
        return min(count / 8.0, 1.0)

    async def handle_discord_message(self, message: Any) -> str | None:
        """Publish human-authored Discord messages as INPUT_RECEIVED events."""

        if not self._publisher:
            logger.warning("DiscordGatewayService publisher unavailable; dropping message")
            return None

        payload = self.build_input_payload(message)
        self._record_inbound_activity(channel_id=payload.channel_id, author_id=payload.author_id)
        if payload.input_id and payload.channel_id:
            self._pending_routes[payload.input_id] = _PendingRoute(
                channel_id=payload.channel_id,
                source_message_id=payload.message_id,
                thread_id=payload.thread_id,
                author_id=payload.author_id,
            )

        await self._publisher.publish(
            EventSubjects.INPUT_RECEIVED,
            payload,
            use_jetstream=True,
            timeout=10.0,
        )
        return payload.input_id

    @asynccontextmanager
    async def _typing_scope(self, channel: _Channel, typing_seconds: float):
        if typing_seconds <= 0:
            yield
            return
        typing = getattr(channel, "typing", None)
        if callable(typing):
            async with typing():
                await self._sleep(typing_seconds)
                yield
            return
        await self._sleep(typing_seconds)
        yield

    async def _wait_for_cooldown(self, channel_id: str, author_id: str | None) -> list[str]:
        now = self._clock()
        keys = [f"channel:{channel_id}"]
        if author_id:
            keys.append(f"user:{author_id}")
        wait = max([self._cooldown_until.get(key, 0.0) - now for key in keys], default=0.0)
        if wait > 0:
            await self._sleep(wait)
        return keys

    def _set_cooldown(self, keys: list[str], cooldown_seconds: float) -> None:
        until = self._clock() + cooldown_seconds
        for key in keys:
            self._cooldown_until[key] = until

    async def _send_channel_message(
        self,
        channel_id: str,
        *,
        content: str,
        reply_to_message_id: str | None,
        thread_id: str | None,
        author_id: str | None,
        interaction_metadata: dict[str, Any] | None,
    ) -> bool:
        if self._discord_client is None:
            logger.warning("Discord client unavailable; cannot forward ranked response")
            return False

        target_id = thread_id or channel_id
        try:
            channel = self._discord_client.get_channel(int(target_id))
        except TypeError:
            logger.warning("Invalid channel id type: %s", target_id)
            return True
        except ValueError:
            logger.warning("Invalid channel id: %s", target_id)
            return True
        if channel is None:
            logger.warning("Channel %s not found", target_id)
            return True

        decision = self._interaction_policy.decide(
            message_text=content,
            channel_pace=self._estimate_channel_pace(channel_id),
            familiarity=self._estimate_familiarity(channel_id, author_id),
            metadata=interaction_metadata,
        )
        cooldown_keys = await self._wait_for_cooldown(channel_id, author_id)
        if decision.delay_seconds > 0:
            await self._sleep(decision.delay_seconds)

        send_kwargs: dict[str, Any] = {}
        if reply_to_message_id is not None:
            send_kwargs["reference"] = reply_to_message_id
        async with self._typing_scope(channel, decision.typing_seconds):
            await channel.send(content, **send_kwargs)
        self._set_cooldown(cooldown_keys, decision.cooldown_seconds)
        return True

    async def _handle_ranked_response(self, msg: Msg) -> None:
        action = "ack"
        try:
            data = json.loads(msg.data.decode())
            if not isinstance(data, dict):
                raise ValueError("ResponseRanked payload must be a dict")
            payload = ResponseRankedPayload.from_dict(data)
            if not payload.final_response:
                raise ValueError("final_response is required")

            route = self._pending_routes.get(payload.input_id or "") if payload.input_id else None
            channel_id = payload.channel_id or (route.channel_id if route else None)
            reply_to_message_id = payload.reply_to_message_id or (route.source_message_id if route else None)
            thread_id = payload.thread_id or (route.thread_id if route else None)
            author_id = payload.author_id or (route.author_id if route else None)
            if not channel_id:
                logger.warning("No channel mapping found for ranked response input_id=%s", payload.input_id)
            else:
                sent = await self._send_channel_message(
                    channel_id,
                    content=payload.final_response,
                    reply_to_message_id=reply_to_message_id,
                    thread_id=thread_id,
                    author_id=author_id,
                    interaction_metadata=payload.interaction_policy,
                )
                if sent:
                    if payload.input_id:
                        self._pending_routes.pop(payload.input_id, None)
                elif hasattr(msg, "nak") and callable(msg.nak):
                    action = "nak"
                else:
                    action = "ack"
        except json.JSONDecodeError:
            logger.error("Invalid ranked response payload", exc_info=True)
            action = "nak"
        except ValueError:
            logger.error("Invalid ranked response payload", exc_info=True)
            action = "nak"
        except Exception:
            logger.error("Failed to handle ranked response", exc_info=True)
            action = "nak"

        if action == "nak" and hasattr(msg, "nak") and callable(msg.nak):
            await msg.nak()
        elif hasattr(msg, "ack") and callable(msg.ack):
            await msg.ack()
