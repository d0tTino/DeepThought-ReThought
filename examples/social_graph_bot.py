import asyncio
import datetime
import json
import logging
import os
import random
import uuid
from datetime import timedelta, timezone
from typing import List, Tuple

import aiohttp

from deepthought.goal_scheduler import GoalScheduler
from deepthought.services import PersonaManager
from deepthought.services.db_manager import (
    MAX_MEMORY_LENGTH,
    MAX_PROMPT_LENGTH,
    MAX_THEORY_LENGTH,
    DBManager,
)
from deepthought.services.moderation import is_allowed
from deepthought.services.scheduler import SchedulerService

try:
    import discord
except Exception:  # pragma: no cover - optional dependency
    from datetime import datetime as dt_datetime
    from datetime import timezone as dt_timezone
    from types import SimpleNamespace

    class _DummyUtils(SimpleNamespace):
        @staticmethod
        def utcnow():
            return dt_datetime.now(dt_timezone.utc)

    class Client:
        def __init__(self, *args, **kwargs):  # pragma: no cover - stub
            pass

        async def close(self):  # pragma: no cover - stub
            return None

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
from nats.js.client import JetStreamContext

SENTIMENT_BACKEND = os.getenv("SENTIMENT_BACKEND", "textblob").lower()
if SENTIMENT_BACKEND == "vader":
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    _sentiment = SentimentIntensityAnalyzer()

    def analyze_sentiment(text: str) -> float:
        """Return the compound sentiment score using VADER."""
        return _sentiment.polarity_scores(text)["compound"]

else:
    from textblob import TextBlob

    def analyze_sentiment(text: str) -> float:
        """Return the sentiment polarity using TextBlob."""
        return TextBlob(text).sentiment.polarity


try:
    from deepthought.config import get_settings
    from deepthought.eda.events import EventSubjects, InputReceivedPayload
    from deepthought.eda.publisher import Publisher
except Exception:  # pragma: no cover - optional dependency
    from types import SimpleNamespace

    def get_settings():
        return SimpleNamespace(
            nats_url="nats://localhost:4222",
            social_graph_db="social_graph.db",
        )

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
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

DB_PATH = get_settings().social_graph_db
CURRENT_DB_PATH = DB_PATH


# Endpoint for forwarding collected data
PRISM_ENDPOINT = os.getenv("PRISM_ENDPOINT", "http://localhost:5000/receive_data")

# NATS configuration for publishing events
NATS_URL = get_settings().nats_url
_nats_client: nats.aio.client.Client | None = None
_js_context: JetStreamContext | None = None
_input_publisher: Publisher | None = None

# Configuration values
MAX_BOT_SPEAKERS = int(os.getenv("MAX_BOT_SPEAKERS", "2"))
IDLE_TIMEOUT_MINUTES = int(os.getenv("IDLE_TIMEOUT_MINUTES", "5"))
PLAYFUL_REPLY_TIMEOUT_MINUTES = int(os.getenv("PLAYFUL_REPLY_TIMEOUT_MINUTES", "5"))
REFLECTION_CHECK_SECONDS = int(os.getenv("REFLECTION_CHECK_SECONDS", "300"))
SENTIMENT_THRESHOLD = float(os.getenv("SENTIMENT_THRESHOLD", "0.3"))

# Optional bot-to-bot chatter configuration
# Accepts values like "true", "1", or "yes" (case-insensitive)
BOT_CHAT_ENABLED = os.getenv("BOT_CHAT_ENABLED", "false").lower() in {
    "true",
    "1",
    "yes",
}

# Candidate prompts used when the bot speaks after a period of silence
idle_response_candidates = [
    "Ever feel like everyone vanished?",
    "I'm still here if anyone wants to chat!",
    "Silence can be golden, but conversation is better.",
]

# Persona-based canned replies for immediate acknowledgements
PERSONA_REPLIES = {
    "friendly": ["I'm pondering your message..."],
    "playful": ["Hmm, let me think on that!"],
    "snarky": ["Yeah, yeah, I'll think about it."],
}

# -----------------------------
# Idle text generation helpers
# -----------------------------
_idle_text_generator = None


def _get_idle_generator():
    """Return a cached HuggingFace text-generation pipeline."""
    global _idle_text_generator
    if _idle_text_generator is None:
        from transformers import pipeline

        model_name = os.getenv("IDLE_MODEL_NAME", "distilgpt2")
        _idle_text_generator = pipeline("text-generation", model=model_name)
    return _idle_text_generator


async def generate_idle_response(prompt: str | None = None) -> str | None:
    """Generate a prompt to send when the channel has been idle.

    The seed text can be provided via ``prompt`` or the ``IDLE_GENERATOR_PROMPT``
    environment variable. ``None`` is returned if generation fails for any
    reason.
    """
    try:
        gen_prompt = prompt or os.getenv("IDLE_GENERATOR_PROMPT", "Say something to spark conversation.")
        if prompt is None and "IDLE_GENERATOR_PROMPT" not in os.environ:
            topics = await get_recent_topics(3)
            if topics:
                gen_prompt = ", ".join(topics) + ": " + gen_prompt

        generator = _get_idle_generator()
        outputs = await asyncio.to_thread(
            generator,
            gen_prompt,
            max_new_tokens=20,
            num_return_sequences=1,
        )

        text = outputs[0]["generated_text"].strip()
        return text
    except Exception:  # pragma: no cover - optional dependency or runtime error
        logger.exception("Idle text generation failed")
        return None


# Simple list of phrases considered bullying
BULLYING_PHRASES = ["idiot", "stupid", "loser", "dumb", "ugly"]

DEFAULT_DB_PATH = DB_PATH
db_manager = DBManager()
persona_manager = PersonaManager(db_manager)


async def init_db(db_path: str | None = None) -> None:
    """Initialize the database, recreating the manager when the path changes."""
    global db_manager, persona_manager, CURRENT_DB_PATH

    target_path = (
        db_path
        if db_path is not None
        else (DB_PATH if DB_PATH != CURRENT_DB_PATH and db_manager.db_path == CURRENT_DB_PATH else db_manager.db_path)
    )

    if db_manager.db_path != target_path:
        if db_manager._db is not None:
            await db_manager.close()
        db_manager = DBManager(target_path)

    await db_manager.init_db()
    persona_manager = PersonaManager(db_manager)
    CURRENT_DB_PATH = db_manager.db_path


async def log_interaction(
    user_id: int,
    target_id: int | None = None,
    sentiment_score: float | None = None,
) -> None:
    await db_manager.log_interaction(user_id, target_id, sentiment_score=sentiment_score)


async def recall_user(user_id: int):
    return await db_manager.recall_user(user_id)


async def store_memory(
    user_id: int,
    memory: str,
    topic: str = "",
    sentiment_score: float | None = None,
) -> None:
    await db_manager.store_memory(user_id, memory, topic=topic, sentiment_score=sentiment_score)


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
    global _nats_client, _js_context, _input_publisher
    if _input_publisher is not None:
        return
    try:
        settings = get_settings()
        _nats_client = await nats.connect(servers=[settings.nats_url])
        _js_context = _nats_client.jetstream()
        _input_publisher = Publisher(_nats_client, _js_context)
    except Exception as exc:  # pragma: no cover - connection issues
        logger.warning("Failed to connect to NATS: %s", exc)
        _input_publisher = None


async def publish_input_received(text: str) -> None:
    """Publish an INPUT_RECEIVED event using NATS JetStream."""
    if not is_allowed(text):  # noqa: F821 - defined in optional module
        logger.info("Dropping INPUT_RECEIVED due to banned content")
        return
    await _ensure_nats()
    if _input_publisher is None:
        logger.warning("Dropping INPUT_RECEIVED event because NATS publisher is unavailable")

        return
    payload = InputReceivedPayload(
        user_input=text,
        input_id=str(uuid.uuid4()),
        timestamp=discord.utils.utcnow().replace(tzinfo=timezone.utc).isoformat(),
    )
    try:
        await _input_publisher.publish(
            EventSubjects.INPUT_RECEIVED,
            payload,
            use_jetstream=True,
            timeout=5.0,
        )
    except Exception as exc:  # pragma: no cover - publish error
        logger.warning("Failed to publish INPUT_RECEIVED: %s", exc)


async def store_theory(subject_id: int, theory: str, confidence: float) -> None:
    return await db_manager.store_theory(subject_id, theory, confidence)


async def get_theories(subject_id: int):
    return await db_manager.get_theories(subject_id)


async def update_sentiment_trend(
    user_id: int,
    channel_id: int,
    sentiment_score: float,
) -> None:
    await db_manager.update_sentiment_trend(user_id, channel_id, sentiment_score)


async def get_sentiment_trend(user_id: int, channel_id: int):
    return await db_manager.get_sentiment_trend(user_id, channel_id)


async def get_recent_topics(limit: int = 3) -> list[str]:
    return await db_manager.get_recent_topics(limit)


async def queue_deep_reflection(user_id: int, context: dict, prompt: str) -> int:
    return await db_manager.queue_deep_reflection(user_id, context, prompt)


async def add_summary_goal(user_id: int, context: dict, prompt: str) -> int:
    """Add a generated summary and goal entry."""
    return await db_manager.add_summary_goal(user_id, context, prompt)


async def list_pending_summary_goals():
    return await db_manager.list_pending_summary_goals()


async def mark_summary_goal_done(task_id: int) -> None:
    await db_manager.mark_summary_goal_done(task_id)


async def set_do_not_mock(user_id: int, flag: bool = True) -> None:
    await db_manager.set_do_not_mock(user_id, flag)


async def is_do_not_mock(user_id: int) -> bool:
    return await db_manager.is_do_not_mock(user_id)


async def adjust_affinity(user_id: int, delta: int) -> None:
    await db_manager.adjust_affinity(user_id, delta)


async def get_affinity(user_id: int) -> int:
    return await db_manager.get_affinity(user_id)


async def get_friendliness(user_id: int, target_id: int) -> float:
    return await db_manager.get_friendliness(user_id, target_id)


async def get_hostility(user_id: int, target_id: int) -> float:
    return await db_manager.get_hostility(user_id, target_id)


async def set_theme(user_id: int, channel_id: int, theme: str) -> None:
    await db_manager.set_theme(user_id, channel_id, theme)


async def get_theme(user_id: int, channel_id: int):
    """Return the last assigned theme for a user/channel pair."""
    return await db_manager.get_theme(user_id, channel_id)


async def assign_themes() -> None:
    """Update the theme for each user/channel based on sentiment trends."""
    rows = await db_manager.get_all_sentiment_trends()
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
        await db_manager.set_theme(user_id, channel_id, theme)


def generate_reflection(prompt: str) -> str:
    """Return a simple reflection string based on sentiment analysis."""
    polarity = analyze_sentiment(prompt)
    if polarity > 0.1:
        mood = "positive"
    elif polarity < -0.1:
        mood = "negative"
    else:
        mood = "neutral"
    return f"Your message felt {mood}."


async def process_deep_reflections(bot: discord.Client) -> None:
    """Background task to process queued reflections."""
    await bot.wait_until_ready()
    while not bot.is_closed():
        try:
            rows = await db_manager.list_pending_tasks()
            if not rows:
                logger.debug("No queued reflections to process")
            for task_id, user_id, ctx_json, prompt in rows:
                context = json.loads(ctx_json)
                channel = bot.get_channel(int(context.get("channel_id")))
                msg_id = context.get("message_id")
                ref = None
                if channel and msg_id:
                    try:
                        ref = await channel.fetch_message(int(msg_id))
                    except Exception:
                        ref = None
                if channel:
                    await asyncio.sleep(2)
                    reflection = generate_reflection(prompt)
                    logger.info(f"Posting deep reflection for task {task_id}")
                    await channel.send(
                        f"After some thought... {reflection}",
                        reference=ref,
                    )
                await db_manager.mark_task_done(task_id)
            await assign_themes()
            await asyncio.sleep(REFLECTION_CHECK_SECONDS)
        except asyncio.CancelledError:
            logger.info("process_deep_reflections cancelled")
            break


async def process_goals(bot: "SocialGraphBot") -> None:
    """Background task that schedules reminders for queued goals."""
    await bot.wait_until_ready()
    while not bot.is_closed():
        try:
            if bot.scheduler_service is None:
                await asyncio.sleep(1)
                continue

            goal = bot.goal_scheduler.next_goal()
            if goal:
                try:
                    delay_str, message = goal.split(":", 1)
                    delay = int(delay_str)
                except ValueError:
                    logger.warning("Invalid goal format: %s", goal)
                else:
                    when = discord.utils.utcnow().replace(tzinfo=timezone.utc) + timedelta(seconds=delay)
                    bot.scheduler_service.schedule_reminder(message, when, str(uuid.uuid4()))
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            logger.info("process_goals cancelled")
            break


def evaluate_triggers(message: discord.Message) -> List[Tuple[str, float]]:
    """Return a list of (theory, confidence) pairs inferred from a message."""
    theories: List[Tuple[str, float]] = []
    if message.created_at.hour == 2:
        theories.append(("insomniac", 0.7))
    lower = message.content.lower()
    if lower.startswith("i agree") or lower.startswith("you're right"):
        theories.append(("social chameleon", 0.6))
    return theories


async def who_is_active(channel: discord.TextChannel, limit: int = 20):
    """Return sets of bot and human authors and bot timestamps from recent messages."""
    bots = set()
    humans = set()
    bot_times: dict[int, datetime.datetime] = {}
    async for msg in channel.history(limit=limit):
        if msg.author.bot:
            bots.add(msg.author.id)
            bot_times.setdefault(msg.author.id, msg.created_at)
        else:
            humans.add(msg.author.id)
    return bots, humans, bot_times


async def last_human_message_age(channel: discord.TextChannel, limit: int = 50):
    """Return minutes since the most recent human message or ``None`` if none."""
    async for msg in channel.history(limit=limit):
        if not msg.author.bot:
            return (discord.utils.utcnow() - msg.created_at.replace(tzinfo=timezone.utc)).total_seconds() / 60
    return None


async def monitor_channels(bot: discord.Client, channel_id: int) -> None:
    """Monitor a channel and occasionally speak during idle periods."""
    await bot.wait_until_ready()
    channel = bot.get_channel(channel_id)
    if channel is None:
        logger.error("Channel %s does not exist", channel_id)
        return
    while not bot.is_closed():
        try:
            last_message = None
            prev_message = None
            idx = 0
            async for msg in channel.history(limit=2):
                if idx == 0:
                    last_message = msg
                elif idx == 1:
                    prev_message = msg
                idx += 1

            respond_to = None
            send_prompt = False
            if last_message and last_message.author.bot and prev_message and not prev_message.author.bot:
                age = (
                    discord.utils.utcnow() - prev_message.created_at.replace(tzinfo=timezone.utc)
                ).total_seconds() / 60
                if age < PLAYFUL_REPLY_TIMEOUT_MINUTES:
                    await asyncio.sleep(60)
                    continue

            if not last_message:

                send_prompt = True
            else:
                idle_minutes = (
                    discord.utils.utcnow() - last_message.created_at.replace(tzinfo=timezone.utc)
                ).total_seconds() / 60
                if idle_minutes >= IDLE_TIMEOUT_MINUTES:
                    send_prompt = True
                elif BOT_CHAT_ENABLED:
                    bots, humans, _ = await who_is_active(channel)
                    if bots and not humans:
                        age = await last_human_message_age(channel)
                        if age is None or age >= PLAYFUL_REPLY_TIMEOUT_MINUTES:
                            send_prompt = True
                            if last_message.author.bot:
                                respond_to = last_message

            if send_prompt:
                prompt = await generate_idle_response()
                if not prompt:
                    prompt = random.choice(idle_response_candidates)
                async with channel.typing():
                    await asyncio.sleep(random.uniform(3, 10))
                    if respond_to is not None:
                        await channel.send(prompt, reference=respond_to)
                    else:
                        await channel.send(prompt)
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            logger.info("monitor_channels cancelled")
            break


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
        self.goal_scheduler = GoalScheduler()
        self.scheduler_service: SchedulerService | None = None  # noqa: F821 - optional feature
        self.persona_manager = PersonaManager(db_manager)

    async def setup_hook(self) -> None:
        await db_manager.connect()
        await init_db()
        self.persona_manager = PersonaManager(db_manager)

        self._bg_tasks.append(self.loop.create_task(monitor_channels(self, self.monitor_channel_id)))
        self._bg_tasks.append(self.loop.create_task(process_deep_reflections(self)))
        self._bg_tasks.append(self.loop.create_task(process_goals(self)))

    async def on_ready(self) -> None:
        """Log basic information once the bot connects."""
        logger.info("Logged in as %s (%s)", self.user.name, self.user.id)

    async def on_message(self, message: discord.Message) -> None:
        if message.author == self.user:
            return

        if any(getattr(m, "bot", False) for m in message.mentions) and self.user not in message.mentions:
            return

        sentiment_score = analyze_sentiment(message.content)
        topic = "message" if abs(sentiment_score) > SENTIMENT_THRESHOLD else ""
        await store_memory(
            message.author.id,
            message.content,
            topic=topic,
            sentiment_score=sentiment_score,
        )
        await update_sentiment_trend(message.author.id, message.channel.id, sentiment_score)

        result = await who_is_active(message.channel)
        if len(result) == 3:
            bots, _, bot_times = result
        else:
            bots, _ = result
            bot_times = {}
        now = discord.utils.utcnow()
        user_id = self.user.id if self.user else None
        for bot_id, ts in bot_times.items():
            if user_id is None or bot_id != user_id:
                age = (now - ts.replace(tzinfo=timezone.utc)).total_seconds() / 60
                if age < PLAYFUL_REPLY_TIMEOUT_MINUTES:
                    return

        if len(bots) > MAX_BOT_SPEAKERS and self.user not in message.mentions:
            # Too many bots talking and we're not addressed directly
            return

        async with message.channel.typing():
            await asyncio.sleep(random.uniform(1, 3))
            if hasattr(message.channel, "history"):
                async for recent in message.channel.history(limit=1):
                    if recent.id != message.id and getattr(recent.author, "bot", False):
                        return
            persona = await self.persona_manager.get_persona(message.author.id)
            reply = random.choice(PERSONA_REPLIES.get(persona, PERSONA_REPLIES["snarky"]))
            await message.channel.send(reply)

        # Log the interaction
        await log_interaction(message.author.id, message.channel.id)

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
        if self.scheduler_service is not None:
            await self.scheduler_service.stop()
            self.scheduler_service = None
        await db_manager.close()
        global _nats_client, _js_context, _input_publisher
        if _nats_client is not None and not _nats_client.is_closed:
            await _nats_client.close()
        _nats_client = None
        _js_context = None
        _input_publisher = None
        await super().close()


def run(token: str, monitor_channel_id: int) -> None:
    """Run the SocialGraphBot."""
    bot = SocialGraphBot(monitor_channel_id=monitor_channel_id)
    bot.run(token)


if __name__ == "__main__":
    from deepthought.config import load_bot_env

    env = load_bot_env()
    run(env.DISCORD_TOKEN, env.MONITOR_CHANNEL)
