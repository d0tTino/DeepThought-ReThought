import asyncio
import json
import logging
import os
import random
import uuid
from datetime import timezone
from typing import List, Tuple

import aiohttp
import aiosqlite

try:
    import discord
except Exception:  # pragma: no cover - optional dependency
    from datetime import datetime
    from datetime import timezone as dt_timezone
    from types import SimpleNamespace

    class _DummyUtils(SimpleNamespace):
        @staticmethod
        def utcnow():
            return datetime.now(dt_timezone.utc)

    class Client:
        async def wait_until_ready(self) -> None:  # pragma: no cover - stub
            return None

        def get_channel(self, _cid):  # pragma: no cover - stub
            return None

        def is_closed(self) -> bool:  # pragma: no cover - stub
            return True

    class Message(SimpleNamespace):  # pragma: no cover - stub
        pass

    class TextChannel(SimpleNamespace):  # pragma: no cover - stub
        async def history(self, *args, **kwargs):
            if False:
                yield  # pragma: no cover - stub

    class Intents(SimpleNamespace):
        @classmethod
        def default(cls):
            return cls()

    discord = SimpleNamespace(
        Client=Client,
        Message=Message,
        TextChannel=TextChannel,
        Intents=Intents,
        utils=_DummyUtils,
    )

import nats
from nats.aio.client import Client as NATS
from nats.js.client import JetStreamContext

from deepthought import social_graph as sg
from deepthought.social_graph import (BOT_CHAT_ENABLED, BULLYING_PHRASES,
                                      CURRENT_DB_PATH, DB_PATH,
                                      IDLE_TIMEOUT_MINUTES, MAX_MEMORY_LENGTH,
                                      MAX_PROMPT_LENGTH, MAX_THEORY_LENGTH,
                                      PLAYFUL_REPLY_TIMEOUT_MINUTES,
                                      REFLECTION_CHECK_SECONDS, DBManager,
                                      SocialGraphService, adjust_affinity,
                                      analyze_sentiment,
                                      generate_idle_response, get_affinity,
                                      get_all_sentiment_trends,
                                      get_recent_topics, get_sentiment_trend,
                                      get_theme, get_theories,
                                      idle_response_candidates, init_db,
                                      is_do_not_mock, log_interaction,
                                      queue_deep_reflection, recall_user,
                                      set_do_not_mock, set_theme, store_memory,
                                      store_theory, update_sentiment_trend)

try:
    from deepthought.config import get_settings
    from deepthought.eda.events import EventSubjects, InputReceivedPayload
    from deepthought.eda.publisher import Publisher
except Exception:  # pragma: no cover - optional dependency
    from types import SimpleNamespace

    def get_settings():
        return SimpleNamespace(nats_url="nats://localhost:4222")

    class EventSubjects(SimpleNamespace):
        INPUT_RECEIVED = "dtr.input.received"

    class InputReceivedPayload:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def to_json(self) -> str:
            return "{}"

    class Publisher:
        def __init__(self, *args, **kwargs) -> None:
            self._nc = None

        async def publish(self, *args, **kwargs) -> None:
            return None


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

PRISM_ENDPOINT = os.getenv("PRISM_ENDPOINT", "http://localhost:5000/receive_data")
NATS_URL = os.getenv("NATS_URL", "nats://localhost:4222")
sg._nats_client = None
sg._js_context = None
sg._input_publisher = None
MAX_BOT_SPEAKERS = int(os.getenv("MAX_BOT_SPEAKERS", "2"))
SENTIMENT_THRESHOLD = float(os.getenv("SENTIMENT_THRESHOLD", "0.3"))


async def send_to_prism(data: dict) -> None:
    """Send collected data to a Prism endpoint."""
    try:
        async with aiohttp.ClientSession() as session:
            await session.post(PRISM_ENDPOINT, json=data, timeout=5)
    except aiohttp.ClientError as exc:
        logger.warning("ClientError sending data to Prism: %s", exc)
    except asyncio.TimeoutError as exc:
        logger.warning("TimeoutError sending data to Prism: %s", exc)
    except Exception as exc:  # pragma: no cover - unexpected errors
        logger.warning("Failed to send data to Prism: %s", exc)


async def _ensure_nats() -> None:
    """Initialize NATS client and publisher if not already connected."""
    if sg._input_publisher is not None:
        return
    try:
        settings = get_settings()
        sg._nats_client = await nats.connect(servers=[settings.nats_url])
        sg._js_context = sg._nats_client.jetstream()
        sg._input_publisher = Publisher(sg._nats_client, sg._js_context)
    except Exception as exc:  # pragma: no cover - connection issues
        logger.warning("Failed to connect to NATS: %s", exc)
        sg._input_publisher = None


async def publish_input_received(text: str) -> None:
    """Publish an INPUT_RECEIVED event using NATS JetStream."""
    await _ensure_nats()
    if sg._input_publisher is None:
        logger.warning(
            "Dropping INPUT_RECEIVED event because NATS publisher is unavailable"
        )

        return
    payload = InputReceivedPayload(
        user_input=text,
        input_id=str(uuid.uuid4()),
        timestamp=discord.utils.utcnow().replace(tzinfo=timezone.utc).isoformat(),
    )
    try:
        await sg._input_publisher.publish(
            EventSubjects.INPUT_RECEIVED,
            payload,
            use_jetstream=True,
            timeout=5.0,
        )
    except Exception as exc:  # pragma: no cover - publish error
        logger.warning("Failed to publish INPUT_RECEIVED: %s", exc)


async def assign_themes() -> None:
    """Update the theme for each user/channel based on sentiment trends."""
    rows = await sg.db_manager.get_all_sentiment_trends()
    for user_id, channel_id, ssum, count in rows:
        if not count:
            continue
        avg = ssum / count
        if avg > 0.2:
            theme = "positive"
        elif avg < -0.2:
            theme = "negative"
        else:
            theme = "neutral"
        await sg.db_manager.set_theme(user_id, channel_id, theme)


def evaluate_triggers(message: discord.Message) -> List[Tuple[str, float]]:
    """Return a list of (theory, confidence) pairs inferred from a message."""
    theories: List[Tuple[str, float]] = []
    if message.created_at.hour == 2:
        theories.append(("insomniac", 0.7))
    lower = message.content.lower()
    if lower.startswith("i agree") or lower.startswith("you're right"):
        theories.append(("social chameleon", 0.6))
    return theories


class SocialGraphBot(discord.Client):
    """Discord bot that records interactions and demonstrates simple awareness."""

    def __init__(self, *args, monitor_channel_id: int, **kwargs):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        intents.presences = True
        super().__init__(*args, intents=intents, **kwargs)
        self.monitor_channel_id = monitor_channel_id
        self._bg_tasks: list[asyncio.Task] = []
        self.service = SocialGraphService(sg.db_manager)

    async def setup_hook(self) -> None:
        await sg.db_manager.connect()
        await init_db()
        self._bg_tasks.append(
            self.loop.create_task(
                self.service.monitor_channels(self, self.monitor_channel_id)
            )
        )
        self._bg_tasks.append(
            self.loop.create_task(self.service.process_deep_reflections(self))
        )

    async def on_ready(self) -> None:
        """Log basic information once the bot connects."""
        logger.info("Logged in as %s (%s)", self.user.name, self.user.id)

    async def on_message(self, message: discord.Message) -> None:
        if message.author == self.user:
            return

        sentiment_score = analyze_sentiment(message.content)
        topic = "message" if abs(sentiment_score) > SENTIMENT_THRESHOLD else ""
        await store_memory(
            message.author.id,
            message.content,
            topic=topic,
            sentiment_score=sentiment_score,
        )
        await update_sentiment_trend(
            message.author.id, message.channel.id, sentiment_score
        )

        bots, _ = await self.service.who_is_active(message.channel)
        if len(bots) > MAX_BOT_SPEAKERS and self.user not in message.mentions:
            return

        # Log the interaction
        await log_interaction(message.author.id, message.channel.id)

        async with message.channel.typing():
            await asyncio.sleep(random.uniform(1, 3))
            if hasattr(message.channel, "history"):
                async for recent in message.channel.history(limit=1):
                    if recent.id != message.id and getattr(recent.author, "bot", False):
                        return
            await message.channel.send("I'm pondering your message...")

        # Publish event and forward to Prism
        await publish_input_received(message.content)

        await send_to_prism(
            {
                "user_id": str(message.author.id),
                "channel_id": str(message.channel.id),
                "content": message.content,
            }
        )

        if any(phrase in message.content.lower() for phrase in BULLYING_PHRASES):
            if not await is_do_not_mock(message.author.id):
                sarcastic = random.choice(
                    [
                        "Oh, how original.",
                        "Wow, such eloquence.",
                        "Tell us how you really feel!",
                    ]
                )
                async with message.channel.typing():
                    await asyncio.sleep(random.uniform(1, 2))
                    await message.channel.send(sarcastic)

        memories = await recall_user(message.author.id)
        if memories:
            logger.info(f"Recalling memories for {message.author.id}: {memories}")

        for theory, conf in evaluate_triggers(message):
            await store_theory(message.author.id, theory, conf)
            await message.channel.send("Some patterns... are best left unspoken.")

        await queue_deep_reflection(
            message.author.id,
            {"channel_id": message.channel.id, "message_id": message.id},
            message.content,
        )

        if hasattr(self, "process_commands"):
            await self.process_commands(message)

    async def close(self) -> None:
        """Cancel background tasks and close external connections."""
        for task in self._bg_tasks:
            task.cancel()
        await asyncio.gather(*self._bg_tasks, return_exceptions=True)
        self._bg_tasks.clear()
        await sg.db_manager.close()
        if sg._nats_client is not None and not sg._nats_client.is_closed:
            await sg._nats_client.close()
        sg._nats_client = None
        sg._js_context = None
        sg._input_publisher = None
        await super().close()


def run(token: str, monitor_channel_id: int) -> None:
    """Run the SocialGraphBot."""
    bot = SocialGraphBot(monitor_channel_id=monitor_channel_id)
    bot.run(token)


if __name__ == "__main__":
    from deepthought.config import load_bot_env

    env = load_bot_env()
    run(env.DISCORD_TOKEN, env.MONITOR_CHANNEL)
