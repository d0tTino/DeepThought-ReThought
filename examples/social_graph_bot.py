import asyncio
import datetime
import json
import logging
import os
import random
import re
import uuid
from collections import deque
from datetime import timedelta, timezone
from typing import Awaitable, Callable, List, Tuple

import aiohttp

from deepthought.bot import deception as bot_deception
from deepthought.bot import interaction as bot_interaction
from deepthought.bot import memory as bot_memory
from deepthought.bot.memory import (
    add_summary_goal,
    adjust_affinity,
    assign_themes,
    get_affinity,
    get_friendliness,
    get_hostility,
    get_interaction_weight,
    get_last_interaction,
    get_pair_mutual_affinity,
    get_recent_topics,
    get_sentiment_trend,
    get_theme,
    get_theories,
    is_do_not_mock,
    list_pending_summary_goals,
    mark_summary_goal_done,
    queue_deep_reflection,
    recall_user,
)
from deepthought.bot.memory import set_db_manager as memory_set_db_manager
from deepthought.bot.memory import (
    set_do_not_mock,
    set_theme,
    store_memory,
    store_theory,
    update_sentiment_trend,
)

# Re-export bot helper functions and configuration values for convenience
maybe_deceptive_reply = bot_deception.maybe_deceptive_reply
store_lie = bot_deception.store_lie
get_last_lie = bot_deception.get_last_lie
deception_set_db_manager = bot_deception.set_db_manager

log_interaction = bot_interaction.log_interaction
interaction_set_db_manager = bot_interaction.set_db_manager

ALLOW_DECEPTION = bot_deception.ALLOW_DECEPTION
DECEPTION_COVER_MESSAGE = bot_deception.DECEPTION_COVER_MESSAGE
DECEPTION_REPLY_MODE = bot_deception.DECEPTION_REPLY_MODE
DYNAMIC_COVER_REPLIES = bot_deception.DYNAMIC_COVER_REPLIES
from deepthought.goal_scheduler import GoalScheduler
from deepthought.perception.emotion_detection import detect_emotions
from deepthought.perception.social_perception import analyze as analyze_social
from deepthought.plugins.projects_board import DEFAULT_HOLIDAY_LOCALE, ProjectsBoard
from deepthought.services import PersonaManager, TrustService
from deepthought.services.db_manager import DBManager
from deepthought.services.manipulative_detection import manipulation_score
from deepthought.services.moderation import evaluate_toxicity, get_profanity_list, is_allowed
from deepthought.services.response_filter import build_response_filter
from deepthought.services.scheduler import SchedulerService
from deepthought.utils import ResponseQueue, UserRateLimiter

try:
    import discord
    from discord.ext import commands
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

    class _DummyCommandTree:
        def __init__(self, client):
            self.client = client

    discord = SimpleNamespace(
        Client=Client,
        Message=Message,
        TextChannel=TextChannel,
        Intents=Intents,
        utils=_DummyUtils,
        app_commands=SimpleNamespace(CommandTree=_DummyCommandTree),
    )

    commands = SimpleNamespace(Bot=Client)

import nats
from nats.aio.msg import Msg
from nats.js.client import JetStreamContext

SENTIMENT_BACKEND = os.getenv("SENTIMENT_BACKEND", "textblob").lower()
if SENTIMENT_BACKEND == "vader":
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        _sentiment = SentimentIntensityAnalyzer()

        def analyze_sentiment(text: str) -> float:
            """Return the compound sentiment score using VADER."""
            return _sentiment.polarity_scores(text)["compound"]

    except Exception:  # pragma: no cover - optional dependency missing
        from textblob import TextBlob

        def analyze_sentiment(text: str) -> float:
            """Fallback to TextBlob sentiment polarity."""
            return TextBlob(text).sentiment.polarity

else:
    from textblob import TextBlob

    def analyze_sentiment(text: str) -> float:
        """Return the sentiment polarity using TextBlob."""
        return TextBlob(text).sentiment.polarity


try:
    from deepthought.config import get_settings
    from deepthought.eda.events import BDIIntentionPayload, EventSubjects, InputReceivedPayload, PlanRequestedPayload
    from deepthought.eda.publisher import Publisher
    from deepthought.eda.subscriber import Subscriber
except Exception:  # pragma: no cover - optional dependency
    from types import SimpleNamespace

    def get_settings():
        return SimpleNamespace(
            nats_url="nats://localhost:4222",
            social_graph_db="social_graph.db",
            persona_descriptions={
                "friendly": "You are a friendly assistant who responds warmly.",
                "playful": "You like to joke around in your answers.",
                "snarky": "You reply with terse, witty sarcasm.",
            },
        )

    class EventSubjects(SimpleNamespace):
        INPUT_RECEIVED = "dtr.input.received"
        PLAN_REQUESTED = "dtr.plan.requested"
        CHAT_RAW = "chat.raw"

    class InputReceivedPayload:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def to_json(self) -> str:
            return "{}"

    class PlanRequestedPayload:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def to_json(self) -> str:
            return "{}"

    class BDIIntentionPayload:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def to_json(self) -> str:
            return "{}"

        @classmethod
        def from_dict(cls, data):
            return cls(**data)

    class Publisher:
        def __init__(self, *args, **kwargs) -> None:
            self._nc = None

        async def publish(self, *args, **kwargs) -> None:
            return None

    class Subscriber:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def subscribe(self, *args, **kwargs) -> None:
            return None

        async def unsubscribe_all(self) -> None:
            return None


logger = logging.getLogger(__name__)

_CHANNEL_MENTION_RE = re.compile(r"<#(\d+)>")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

DB_PATH = get_settings().social_graph_db
CURRENT_DB_PATH = DB_PATH
SOCIAL_PERCEPTION_MODEL = get_settings().social_perception_model


# Endpoint for forwarding collected data
PRISM_ENDPOINT = os.getenv("PRISM_ENDPOINT", "http://localhost:5000/receive_data")

# NATS configuration for publishing events
NATS_URL = get_settings().nats_url
_nats_client: nats.aio.client.Client | None = None
_js_context: JetStreamContext | None = None
_input_publisher: Publisher | None = None
_subscriber: Subscriber | None = None

# Configuration values
MAX_BOT_SPEAKERS = int(os.getenv("MAX_BOT_SPEAKERS", "2"))
IDLE_TIMEOUT_MINUTES = int(os.getenv("IDLE_TIMEOUT_MINUTES", "5"))
PLAYFUL_REPLY_TIMEOUT_MINUTES = int(os.getenv("PLAYFUL_REPLY_TIMEOUT_MINUTES", "5"))
REFLECTION_CHECK_SECONDS = int(os.getenv("REFLECTION_CHECK_SECONDS", "300"))
INTENTION_PUBLISH_SECONDS = int(os.getenv("INTENTION_PUBLISH_SECONDS", "5"))
SENTIMENT_THRESHOLD = float(os.getenv("SENTIMENT_THRESHOLD", "0.3"))
AFFINITY_POS_DELTA = int(os.getenv("AFFINITY_POS_DELTA", "1"))
AFFINITY_NEG_DELTA = int(os.getenv("AFFINITY_NEG_DELTA", "-1"))
USER_REPLY_RATE_SECONDS = float(os.getenv("USER_REPLY_RATE_SECONDS", "3"))
BOT_COOLDOWN_SECONDS = int(os.getenv("BOT_COOLDOWN_SECONDS", "30"))
OTHER_BOT_COOLDOWN_SECONDS = int(os.getenv("OTHER_BOT_COOLDOWN_SECONDS", "10"))
BOT_MESSAGE_INTERVAL_SECONDS = int(os.getenv("BOT_MESSAGE_INTERVAL_SECONDS", "60"))
MAX_BOT_MESSAGES_PER_INTERVAL = int(os.getenv("MAX_BOT_MESSAGES_PER_INTERVAL", "5"))
RESPONSE_QUEUE_COOLDOWN_SECONDS = float(
    os.getenv("RESPONSE_QUEUE_COOLDOWN_SECONDS", str(BOT_MESSAGE_INTERVAL_SECONDS))
)
RESPONSE_STALE_SECONDS = float(os.getenv("RESPONSE_STALE_SECONDS", "20"))
MINIMAL_REPLY_THRESHOLD = float(os.getenv("MINIMAL_REPLY_THRESHOLD", "-5"))
MINIMAL_REPLY_PROB = float(os.getenv("MINIMAL_REPLY_PROB", "0.05"))
MINIMAL_REPLIES = ["...", "👍", "No"]
AVOIDANCE_REPLY = "Take your time; I'm here if you need me."
REFUSAL_MESSAGE = "I'm sorry, but I can't help with that."
FOE_MODE_CAP_ENABLED = os.getenv("FOE_MODE_CAP_ENABLED", "true").lower() in {
    "true",
    "1",
    "yes",
}
FOE_MODE_MAX_SEVERITY = float(os.getenv("FOE_MODE_MAX_SEVERITY", "0.4"))

# Optional channel for thought logging
_THOUGHT_CHANNEL = os.getenv("THOUGHT_CHANNEL")
try:
    THOUGHT_CHANNEL_ID = int(_THOUGHT_CHANNEL) if _THOUGHT_CHANNEL else None
except ValueError:
    THOUGHT_CHANNEL_ID = None

# Comma-separated list of developer IDs allowed to run admin commands
DEVELOPER_IDS = {
    int(uid)
    for uid in os.getenv("DEVELOPER_IDS", "").split(",")
    if uid.strip().isdigit()
}

# Optional bot-to-bot chatter configuration
# Accepts values like "true", "1", or "yes" (case-insensitive)
BOT_CHAT_ENABLED = os.getenv("BOT_CHAT_ENABLED", "false").lower() in {
    "true",
    "1",
    "yes",
}

# Comma-separated list of channel IDs where reply mentions should be suppressed
NO_MENTION_CHANNEL_IDS = {
    int(cid)
    for cid in os.getenv("NO_MENTION_CHANNEL_IDS", "").split(",")
    if cid.strip().isdigit()
}

# Literal handshake message exchanged between bots before chatting.
HANDSHAKE_MESSAGE = "BOT_HANDSHAKE"


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

# Mapping of dominant emotions to reply strategies.  Each entry maps an
# emotion label returned by ``detect_emotions`` to either a persona in
# ``PERSONA_REPLIES`` or the special value ``"minimal"`` indicating the bot
# should respond with ``MINIMAL_REPLIES``.
EMOTION_REPLY_MAP = {
    "Happy": ("persona", "playful"),
    "Surprise": ("persona", "playful"),
    "Sad": ("persona", "friendly"),
    "Angry": ("minimal", None),
    "Fear": ("minimal", None),
}

# -----------------------------
# Idle text generation helpers
# -----------------------------
_idle_text_generator = None
_response_text_generator = None


def _get_idle_generator():
    """Return a cached HuggingFace text-generation pipeline."""
    global _idle_text_generator
    if _idle_text_generator is None:
        from transformers import pipeline

        model_name = os.getenv("IDLE_MODEL_NAME", "distilgpt2")
        _idle_text_generator = pipeline("text-generation", model=model_name)
    return _idle_text_generator


async def _generate_text(prompt: str, *, max_new_tokens: int) -> str | None:
    """Return generated text for ``prompt`` or ``None`` on failure."""
    try:
        generator = _get_idle_generator()
        outputs = await asyncio.to_thread(
            generator,
            prompt,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
        )
        text = outputs[0]["generated_text"].strip()
        return text
    except Exception:  # pragma: no cover - optional dependency or runtime error
        logger.exception("Text generation failed")
        return None


def _build_persona_prompt(
    messages: List[str], persona_desc: str, memory_snippets: list[str] | None = None
) -> str:
    """Return a prompt that includes persona, memory context, and message history."""

    history = "\n".join(messages)
    sections = []
    if persona_desc:
        sections.append(persona_desc)
    if memory_snippets:
        memory_lines = "\n".join(f"- {snippet}" for snippet in memory_snippets)
        sections.append(f"Memory notes:\n{memory_lines}")
    sections.append(history)
    return "\n\n".join(sections)


async def generate_idle_response(prompt: str | None = None) -> str | None:
    """Generate a prompt to send when the channel has been idle.

    The seed text can be provided via ``prompt`` or the ``IDLE_GENERATOR_PROMPT``
    environment variable. ``None`` is returned if generation fails for any
    reason.
    """
    gen_prompt = prompt or os.getenv("IDLE_GENERATOR_PROMPT", "Say something to spark conversation.")
    if prompt is None and "IDLE_GENERATOR_PROMPT" not in os.environ:
        topics = await get_recent_topics(3)
        if topics:
            gen_prompt = ", ".join(topics) + ": " + gen_prompt

    return await _generate_text(gen_prompt, max_new_tokens=20)


MAX_MEMORY_PROMPT_SNIPPETS = int(os.getenv("MEMORY_PROMPT_SNIPPETS", "5"))
LLM_REPLY_MAX_NEW_TOKENS = int(os.getenv("LLM_REPLY_MAX_NEW_TOKENS", "40"))


def _format_memory_snippets(memories: list[tuple[str, str]], limit: int) -> list[str]:
    if limit <= 0:
        return []
    snippets = []
    for topic, memory in memories:
        if not memory or not str(memory).strip():
            continue
        entry = f"[{topic}] {memory}" if topic else str(memory)
        snippets.append(entry.strip())
    return snippets[-limit:]


def _build_memory_hint(memory_snippets: list[str]) -> str:
    """Return a short, user-facing memory hint."""

    if not memory_snippets:
        return ""
    summary = re.sub(r"\s+", " ", memory_snippets[-1]).strip()
    if len(summary) > 120:
        summary = summary[:117] + "..."
    return f"You mentioned {summary} earlier."


def _append_memory_hint(reply: str, memory_hint: str) -> str:
    """Append memory hint to reply with proper spacing."""

    if not memory_hint:
        return reply
    if not reply:
        return memory_hint
    if reply.endswith((".", "!", "?")):
        return f"{reply} {memory_hint}"
    return f"{reply}. {memory_hint}"


def build_reply_prompt(user_text: str, memory_snippets: list[str]) -> str:
    """Assemble a reply prompt with optional memory context."""
    lines = ["You are a conversational social bot.", "MEMORY_RETRIEVED:"]
    if memory_snippets:
        lines.extend(f"- {snippet}" for snippet in memory_snippets)
    else:
        lines.append("- (none)")
    lines.extend(
        [
            "Respond naturally and briefly.",
            f"User: {user_text}",
            "Bot:",
        ]
    )
    return "\n".join(lines)


async def generate_contextual_reply(user_text: str, memory_snippets: list[str]) -> str | None:
    prompt = build_reply_prompt(user_text, memory_snippets)
    return await _generate_text(prompt, max_new_tokens=LLM_REPLY_MAX_NEW_TOKENS)


async def generate_llm_reply(prompt: str) -> str | None:
    """Generate an LLM reply for ``prompt`` using the contextual generator."""

    return await _generate_text(prompt, max_new_tokens=LLM_REPLY_MAX_NEW_TOKENS)


# Simple list of phrases considered bullying
BULLYING_PHRASES = ["idiot", "stupid", "loser", "dumb", "ugly"]
BULLYING_RESPONSE = "I'm not here to be disrespected. Let's keep things civil."


DEFAULT_DB_PATH = DB_PATH
db_manager = DBManager()
memory_set_db_manager(lambda: db_manager)
deception_set_db_manager(lambda: db_manager)
interaction_set_db_manager(lambda: db_manager)
# Refresh exported configuration after registering DB manager
ALLOW_DECEPTION = bot_deception.ALLOW_DECEPTION
DECEPTION_COVER_MESSAGE = bot_deception.DECEPTION_COVER_MESSAGE
DECEPTION_REPLY_MODE = bot_deception.DECEPTION_REPLY_MODE
DYNAMIC_COVER_REPLIES = bot_deception.DYNAMIC_COVER_REPLIES
persona_manager = PersonaManager(db_manager, descriptions=get_settings().persona_descriptions)
trust_service = TrustService(db_manager)
reply_limiter = UserRateLimiter(1, USER_REPLY_RATE_SECONDS)
bot_last_messages: dict[int, tuple[str, datetime.datetime]] = {}
last_bot_reply_time: datetime.datetime | None = None
bot_message_times: dict[int, deque[datetime.datetime]] = {}
our_message_times: deque[datetime.datetime] = deque()
last_other_bot_message_time: datetime.datetime | None = None

# Track handshake completions and per-bot cooldowns
bot_handshakes: dict[int, datetime.datetime] = {}
bot_reply_times: dict[int, datetime.datetime] = {}


async def init_db(db_path: str | None = None) -> None:
    """Initialize the database, recreating the manager when the path changes."""
    global db_manager, persona_manager, trust_service, CURRENT_DB_PATH

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
    persona_manager = PersonaManager(db_manager, descriptions=get_settings().persona_descriptions)
    trust_service = TrustService(db_manager)
    CURRENT_DB_PATH = db_manager.db_path


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
    """Initialize NATS client, publisher and subscriber if needed."""
    global _nats_client, _js_context, _input_publisher, _subscriber
    if _input_publisher is not None and _subscriber is not None:
        return
    try:
        settings = get_settings()
        _nats_client = await nats.connect(servers=[settings.nats_url])
        _js_context = _nats_client.jetstream()
        _input_publisher = Publisher(_nats_client, _js_context)
        _subscriber = Subscriber(_nats_client, _js_context)
    except Exception as exc:  # pragma: no cover - connection issues
        logger.warning("Failed to connect to NATS: %s", exc)
        _input_publisher = None
        _subscriber = None


async def _send_thought(bot: discord.Client, text: str) -> None:
    """Send ``text`` to the THOUGHT_CHANNEL if configured."""
    if THOUGHT_CHANNEL_ID is None:
        return
    channel = bot.get_channel(THOUGHT_CHANNEL_ID)
    if channel is None:
        logger.warning("Thought channel %s not found", THOUGHT_CHANNEL_ID)
        return
    if hasattr(bot, "response_filter"):
        filtered = bot.response_filter.sanitize(text)  # type: ignore[attr-defined]
        if filtered is None:
            logger.info("Thought message filtered; not sending")
            return
        text = filtered
    try:
        await channel.send(text)
    except Exception as exc:  # pragma: no cover - send failure
        logger.warning("Failed to send thought: %s", exc)


def log_thought(bot: discord.Client, text: str) -> None:
    """Schedule ``text`` to be sent to the THOUGHT_CHANNEL asynchronously."""
    if THOUGHT_CHANNEL_ID is None:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:  # pragma: no cover - no loop available
        return
    loop.create_task(_send_thought(bot, text))


async def emit_refusal(
    message: discord.Message,
    enqueue: Callable[[discord.abc.Messageable, Callable[[], Awaitable[None]]], Awaitable[None]] | None = None,
) -> None:
    """Send a refusal message to ``message.channel``."""

    async def _send_refusal() -> None:
        async with message.channel.typing():
            await asyncio.sleep(random.uniform(1, 3))
            await message.channel.send(REFUSAL_MESSAGE)

    if enqueue is None:
        await _send_refusal()
    else:
        await enqueue(message.channel, _send_refusal)


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


async def publish_plan_requested(goal: str, input_id: str | None = None) -> None:
    """Publish a PLAN_REQUESTED event for ``goal``."""
    await _ensure_nats()
    if _input_publisher is None:
        logger.warning("Dropping PLAN_REQUESTED event because NATS publisher is unavailable")
        return
    payload = PlanRequestedPayload(goal=goal, input_id=input_id)
    try:
        await _input_publisher.publish(
            EventSubjects.PLAN_REQUESTED,
            payload,
            use_jetstream=True,
            timeout=5.0,
        )
    except Exception as exc:  # pragma: no cover - publish error
        logger.warning("Failed to publish PLAN_REQUESTED: %s", exc)


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
                    text = f"After some thought... {reflection}"
                    if hasattr(bot, "response_filter"):
                        text = bot.response_filter.sanitize(text)  # type: ignore[attr-defined]
                    if text:
                        await channel.send(
                            text,
                            reference=ref,
                        )
                await db_manager.mark_task_done(task_id)
            await assign_themes()
            await asyncio.sleep(REFLECTION_CHECK_SECONDS)
        except asyncio.CancelledError:
            logger.info("process_deep_reflections cancelled")
            break


async def _maybe_post_reminder_to_thread(bot: "SocialGraphBot", message: str) -> None:
    """Attempt to echo reminder ``message`` to a referenced project thread."""

    match = _CHANNEL_MENTION_RE.search(message)
    if match is None:
        return

    try:
        thread_id = int(match.group(1))
    except (TypeError, ValueError):
        return

    channel = None
    get_channel = getattr(bot, "get_channel", None)
    if callable(get_channel):
        channel = get_channel(thread_id)

    if channel is None:
        fetch_channel = getattr(bot, "fetch_channel", None)
        if callable(fetch_channel):
            try:
                channel = await fetch_channel(thread_id)
            except Exception:
                channel = None

    if channel is None or not hasattr(channel, "send"):
        return

    try:
        if hasattr(bot, "response_filter"):
            message = bot.response_filter.sanitize(message)  # type: ignore[attr-defined]
        if message:
            await channel.send(message)
    except Exception:
        logger.debug("Failed to post reminder to thread %s", thread_id)


async def process_goals(bot: "SocialGraphBot") -> None:
    """Background task that schedules reminders for queued goals."""
    await bot.wait_until_ready()
    while not bot.is_closed():
        try:
            goal = bot.goal_scheduler.next_goal()
            if goal:
                try:
                    delay_str, message = goal.split(":", 1)
                    delay = int(delay_str)
                except ValueError:
                    logger.warning("Invalid goal format: %s", goal)
                    await publish_plan_requested(goal)
                else:
                    if bot.scheduler_service is not None:
                        when = discord.utils.utcnow().replace(tzinfo=timezone.utc) + timedelta(seconds=delay)
                        bot.scheduler_service.schedule_reminder(message, when, str(uuid.uuid4()))
                        await publish_plan_requested(message)
                    else:
                        # When the scheduler service is unavailable we still emit the reminder
                        # immediately so the project thread continues to receive updates.
                        await publish_plan_requested(message)
                        await _maybe_post_reminder_to_thread(bot, message)
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            logger.info("process_goals cancelled")
            break


async def process_intentions(bot: "SocialGraphBot") -> None:
    """Background task that publishes stored intentions."""
    await bot.wait_until_ready()
    while not bot.is_closed():
        try:
            await bot.goal_scheduler.load_pending_intentions()
            if _input_publisher is not None:
                await bot.goal_scheduler.publish_intentions(_input_publisher)
            await asyncio.sleep(INTENTION_PUBLISH_SECONDS)
        except asyncio.CancelledError:
            logger.info("process_intentions cancelled")
            break


async def process_thought_commands(bot: "SocialGraphBot") -> None:
    """Listen for user commands in the thought channel without responding."""
    if THOUGHT_CHANNEL_ID is None:
        return
    await bot.wait_until_ready()
    check = (
        lambda m: m.channel.id == THOUGHT_CHANNEL_ID and not getattr(m.author, "bot", False) and m.author != bot.user
    )
    while not bot.is_closed():
        try:
            message = await bot.wait_for("message", check=check)
            if message.author.id not in DEVELOPER_IDS:
                continue
            content = message.content.strip()
            if content.startswith("/goal"):
                goal = content.removeprefix("/goal").strip()
                if goal:
                    await bot.goal_scheduler.queue_intention(goal, priority=1)
            elif content.startswith("/memory"):
                mem = content.removeprefix("/memory").strip()
                if mem:
                    await store_memory(message.author.id, mem)
            elif content.startswith("/thought"):
                parts = content.split(maxsplit=3)
                if len(parts) < 2:
                    continue
                cmd = parts[1]
                if cmd == "list":
                    limit = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 5
                    msgs = []
                    async for msg in message.channel.history(limit=limit):
                        if msg.author == bot.user:
                            msgs.append(f"{msg.id}: {msg.content}")
                    if msgs:
                        await bot.send_filtered(
                            message.channel, "\n".join(reversed(msgs))
                        )
                elif cmd == "delete" and len(parts) > 2:
                    try:
                        msg_id = int(parts[2])
                        target = await message.channel.fetch_message(msg_id)
                    except Exception:
                        continue
                    if target.author == bot.user:
                        await target.delete()
                        logger.info("User %s deleted thought %s", message.author.id, msg_id)
                elif cmd == "edit" and len(parts) > 3:
                    try:
                        msg_id = int(parts[2])
                        new_content = parts[3]
                        target = await message.channel.fetch_message(msg_id)
                    except Exception:
                        continue
                    if target.author == bot.user:
                        await target.edit(content=new_content)
                        logger.info("User %s edited thought %s", message.author.id, msg_id)
            elif content.startswith("/lies"):
                parts = content.split(maxsplit=3)
                if len(parts) < 2:
                    continue
                cmd = parts[1]
                if cmd == "list":
                    limit = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 5
                    rows = await db_manager.list_lies(limit)
                    if rows:
                        lines = [f"{rowid}: {question} -> {reply} (user {uid})" for rowid, uid, question, reply in rows]
                        await bot.send_filtered(message.channel, "\n".join(lines))
                elif cmd == "delete" and len(parts) > 2:
                    if parts[2].isdigit():
                        rowid = int(parts[2])
                        removed = await db_manager.delete_lie(rowid)
                        if removed:
                            logger.info("User %s deleted lie %s", message.author.id, rowid)
                elif cmd == "edit" and len(parts) > 3:
                    if parts[2].isdigit():
                        rowid = int(parts[2])
                        new_reply = parts[3]
                        updated = await db_manager.update_lie(rowid, new_reply)
                        if updated:
                            logger.info("User %s edited lie %s", message.author.id, rowid)
        except asyncio.CancelledError:
            logger.info("process_thought_commands cancelled")
            break
        except Exception:  # pragma: no cover - defensive
            logger.exception("Error processing thought command")


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
    """Return sets of bot and human authors and bot timestamps from recent messages.

    Automated bots that repeat the same content are ignored so the caller can
    gauge actual activity. ``author.bot`` is used to separate humans from bots.
    """

    bots = set()
    humans = set()
    bot_times: dict[int, datetime.datetime] = {}
    seen_bot_messages: dict[int, str] = {}

    async for msg in channel.history(limit=limit):
        author = msg.author
        if getattr(author, "bot", False):
            if seen_bot_messages.get(author.id) == msg.content:
                # Skip repeated automated post from the same bot
                continue
            bots.add(author.id)
            bot_times.setdefault(author.id, msg.created_at)
            seen_bot_messages[author.id] = msg.content
        else:
            humans.add(author.id)

    return bots, humans, bot_times


async def last_human_message_age(channel: discord.TextChannel, limit: int = 50):
    """Return minutes since the most recent human message or ``None`` if none."""
    async for msg in channel.history(limit=limit):
        if not msg.author.bot:
            return (discord.utils.utcnow() - msg.created_at.replace(tzinfo=timezone.utc)).total_seconds() / 60
    return None


async def handle_bot_handshake(message: discord.Message) -> bool:
    """Handle bot handshakes and cooldowns.

    Returns ``True`` when normal processing should continue. When a handshake
    message is received the bot echoes ``HANDSHAKE_MESSAGE`` and returns
    ``False`` to stop further processing. Messages from bots that have not
    completed the handshake or are still in their cooldown period are ignored.
    """

    if not BOT_CHAT_ENABLED or not message.author.bot:
        return True

    global last_bot_reply_time
    now = discord.utils.utcnow()
    content = message.content.strip()

    if content == HANDSHAKE_MESSAGE:
        ts = bot_handshakes.get(message.author.id)
        if ts is None or (now - ts).total_seconds() > BOT_COOLDOWN_SECONDS:
            bot_handshakes[message.author.id] = now
            async with message.channel.typing():
                await asyncio.sleep(random.uniform(1, 3))
                await message.channel.send(HANDSHAKE_MESSAGE)
            bot_reply_times[message.author.id] = now
            last_bot_reply_time = now
        return False

    if message.author.id not in bot_handshakes:
        return False

    last = bot_reply_times.get(message.author.id)
    if last and (now - last).total_seconds() < BOT_COOLDOWN_SECONDS:
        return False

    bot_reply_times[message.author.id] = now
    return True


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
                    if bots and not humans and len(bots) < MAX_BOT_SPEAKERS:
                        age = await last_human_message_age(channel)
                        if age is None or age >= PLAYFUL_REPLY_TIMEOUT_MINUTES:
                            send_prompt = True
                            if last_message.author.bot:
                                respond_to = last_message

            if send_prompt:
                prompt = await generate_idle_response()
                if not prompt:
                    prompt = random.choice(idle_response_candidates)
                async def _send_prompt() -> None:
                    async with channel.typing():
                        await asyncio.sleep(random.uniform(3, 10))
                        safe_prompt = (
                            bot.response_filter.sanitize(prompt)
                            if hasattr(bot, "response_filter")
                            else prompt
                        )
                        if not safe_prompt:
                            return
                        if respond_to is not None:
                            await channel.send(safe_prompt, reference=respond_to)
                        else:
                            await channel.send(safe_prompt)

                enqueue = getattr(bot, "enqueue_response", None)
                if callable(enqueue):
                    await enqueue(channel, _send_prompt)
                else:
                    await _send_prompt()
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            logger.info("monitor_channels cancelled")
            break


class SocialGraphBot(commands.Bot):
    """Discord bot that records interactions and demonstrates simple awareness."""

    def __init__(
        self,
        *args,
        monitor_channel_id: int,
        forum_channel_id: int | None = None,
        index_channel_id: int | None = None,
        require_scheduled_events: bool = False,
        holiday_locale: str = DEFAULT_HOLIDAY_LOCALE,
        command_prefix: str = "!",
        **kwargs,
    ):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        intents.presences = True
        super().__init__(*args, command_prefix=command_prefix, intents=intents, **kwargs)
        if not hasattr(self, "tree"):
            self.tree = discord.app_commands.CommandTree(self)
        self.monitor_channel_id = monitor_channel_id
        self.forum_channel_id = forum_channel_id
        self.index_channel_id = index_channel_id
        self.require_scheduled_events = require_scheduled_events
        self.holiday_locale = holiday_locale
        self._bg_tasks: list[asyncio.Task] = []
        self.goal_scheduler = GoalScheduler(db_manager)
        self.scheduler_service: SchedulerService | None = None  # noqa: F821 - optional feature
        self.persona_manager = PersonaManager(db_manager, descriptions=get_settings().persona_descriptions)
        self._subscriber: Subscriber | None = None
        self._projects_board: ProjectsBoard | None = None
        self.response_queue = ResponseQueue(
            RESPONSE_QUEUE_COOLDOWN_SECONDS, stale_after=RESPONSE_STALE_SECONDS
        )
        self.response_filter = build_response_filter(
            denylist=get_profanity_list(), toxicity_threshold=FOE_MODE_MAX_SEVERITY
        )

    async def setup_hook(self) -> None:
        await db_manager.connect()
        await init_db()
        self.persona_manager = PersonaManager(db_manager, descriptions=get_settings().persona_descriptions)

        await _ensure_nats()
        if _nats_client is not None and _js_context is not None:
            try:
                self._subscriber = Subscriber(_nats_client, _js_context)
                await self._subscriber.subscribe(
                    subject=EventSubjects.CHAT_RAW,
                    handler=self._handle_chat_raw,
                    use_jetstream=True,
                    durable="social_bot_chat",
                )
                await self._subscriber.subscribe(
                    subject=EventSubjects.BDI_INTENTION,
                    handler=self._handle_bdi_intention,
                    use_jetstream=True,
                    durable="social_bot_intention",
                )
            except Exception as exc:  # pragma: no cover - subscription error
                logger.warning("Failed to subscribe to CHAT_RAW: %s", exc)
                self._subscriber = None

        self._projects_board = ProjectsBoard(
            self,
            db_manager,
            self.goal_scheduler,
            forum_channel_id=self.forum_channel_id,
            index_channel_id=self.index_channel_id,
            monitor_channel_id=self.monitor_channel_id,
            require_events=self.require_scheduled_events,
            holiday_locale=self.holiday_locale,
        )
        await self.add_cog(self._projects_board)

        self._bg_tasks.append(asyncio.create_task(monitor_channels(self, self.monitor_channel_id)))
        self._bg_tasks.append(asyncio.create_task(process_deep_reflections(self)))
        self._bg_tasks.append(asyncio.create_task(process_goals(self)))
        self._bg_tasks.append(asyncio.create_task(process_intentions(self)))
        if THOUGHT_CHANNEL_ID is not None:
            self._bg_tasks.append(asyncio.create_task(process_thought_commands(self)))

    async def on_ready(self) -> None:
        """Log basic information once the bot connects."""
        logger.info("Logged in as %s (%s)", self.user.name, self.user.id)

    async def enqueue_response(
        self, channel: discord.abc.Messageable, sender: Callable[[], Awaitable[None]]
    ) -> None:
        """Enqueue a response for ``channel`` while respecting cooldowns."""

        await self.response_queue.enqueue(channel.id, sender)

    def sanitize_response(self, text: str) -> str | None:
        return self.response_filter.sanitize(text)

    async def send_filtered(self, channel: discord.abc.Messageable, text: str, **kwargs) -> None:
        sanitized = self.sanitize_response(text)
        if sanitized is None:
            logger.info("Response filtered; nothing sent")
            return
        await channel.send(sanitized, **kwargs)

    async def on_message(self, message: discord.Message) -> None:
        global last_bot_reply_time, last_other_bot_message_time
        if message.author == self.user:
            return

        if getattr(message.author, "bot", False):
            last_other_bot_message_time = message.created_at.replace(tzinfo=timezone.utc)

        if THOUGHT_CHANNEL_ID is not None and message.channel.id == THOUGHT_CHANNEL_ID:
            return

        if any(getattr(m, "bot", False) for m in message.mentions) and self.user not in message.mentions:
            return

        if not await handle_bot_handshake(message):
            return

        now = discord.utils.utcnow()
        if not message.author.bot:
            if (
                last_other_bot_message_time
                and (now - last_other_bot_message_time).total_seconds() < OTHER_BOT_COOLDOWN_SECONDS
            ):
                return

        if message.author.bot:
            q = bot_message_times.setdefault(message.author.id, deque())
            q.append(now)
            while q and (now - q[0]).total_seconds() > BOT_MESSAGE_INTERVAL_SECONDS:
                q.popleft()
            if len(q) >= MAX_BOT_MESSAGES_PER_INTERVAL:
                return
            last = bot_last_messages.get(message.author.id)
            if last and last[0] == message.content and (now - last[1]).total_seconds() < BOT_COOLDOWN_SECONDS:
                return
            bot_last_messages[message.author.id] = (message.content, now)
            if last_bot_reply_time and (now - last_bot_reply_time).total_seconds() < BOT_COOLDOWN_SECONDS:
                return

        if not is_allowed(message.content):  # noqa: F821 - optional import
            await emit_refusal(message, enqueue=self.enqueue_response)
            await trust_service.penalize_banned(message.author.id)
            return

        lie_reply = await maybe_deceptive_reply(message.author.id, message.content)
        if lie_reply:
            if DECEPTION_REPLY_MODE == "dynamic" and lie_reply != DECEPTION_COVER_MESSAGE:
                cover_reply = random.choice(DYNAMIC_COVER_REPLIES)
                await store_lie(message.author.id, message.content, cover_reply)
            else:
                cover_reply = lie_reply
            cover_reply = self.sanitize_response(cover_reply) or None
            if cover_reply is None:
                logger.info("Cover reply blocked by response filter")
                return

            async def send_cover_reply() -> None:
                async with message.channel.typing():
                    await asyncio.sleep(random.uniform(1, 3))
                    await self.send_filtered(message.channel, cover_reply)

            await self.enqueue_response(message.channel, send_cover_reply)
            await store_memory(message.author.id, cover_reply, topic="deception")
            await log_interaction(message.author.id, message.channel.id)
            await publish_input_received(message.content)
            await send_to_prism(
                {
                    "user_id": str(message.author.id),
                    "channel_id": str(message.channel.id),
                    "content": message.content,
                }
            )

            return

        sentiment_score = analyze_sentiment(message.content)
        emotions = detect_emotions(message.content)
        dominant_emotion, dom_score = (None, 0.0)
        if emotions:
            dominant_emotion, dom_score = max(emotions.items(), key=lambda kv: kv[1])
            if dom_score == 0:
                dominant_emotion = None
        log_thought(
            self,
            f"Sentiment score: {sentiment_score:+.2f}, Emotion: {dominant_emotion or 'Neutral'}",
        )
        topic = "message" if abs(sentiment_score) > SENTIMENT_THRESHOLD else ""
        await store_memory(
            message.author.id,
            message.content,
            topic=topic,
            sentiment_score=sentiment_score,
        )
        await update_sentiment_trend(message.author.id, message.channel.id, sentiment_score)
        affinity_delta = None
        if sentiment_score >= SENTIMENT_THRESHOLD:
            affinity_delta = AFFINITY_POS_DELTA
        elif sentiment_score <= -SENTIMENT_THRESHOLD:
            affinity_delta = AFFINITY_NEG_DELTA
        if affinity_delta:
            await adjust_affinity(message.author.id, affinity_delta)

        await db_manager.record_emotion(message.author.id, emotions)

        social_scores = analyze_social(message.content)
        bullying = manipulation_score(message.content, {"bullying": BULLYING_PHRASES})
        manip_category = manipulation_score(message.content)
        category_to_log = manip_category or bullying or max(social_scores, key=social_scores.get)
        await db_manager.record_manipulation(message.author.id, category_to_log)
        if manip_category or social_scores.get("manipulation", 0) > 0.5:
            await trust_service.penalize_manipulative(message.author.id)

            log_thought(self, f"Manipulation detected: {category_to_log}")
            return

        trust = await trust_service.get_trust(message.author.id)
        if trust <= trust_service.lower_limit:
            return

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
        for bot_id, times in bot_message_times.items():
            while times and (now - times[0]).total_seconds() > BOT_MESSAGE_INTERVAL_SECONDS:
                times.popleft()
            if (
                bot_id != message.author.id
                and self.user not in message.mentions
                and (user_id is None or bot_id != user_id)
                and times
                and (now - times[-1]).total_seconds() < BOT_MESSAGE_INTERVAL_SECONDS
            ):
                return

        if not reply_limiter.allow(str(message.author.id)):
            return

        memories = await recall_user(message.author.id)
        memory_snippets = _format_memory_snippets(memories, MAX_MEMORY_PROMPT_SNIPPETS)
        memory_hint = _build_memory_hint(memory_snippets)
        if memories:
            logger.info(f"Recalling memories for {message.author.id}: {memories}")

        async with message.channel.typing():
            start_time = discord.utils.utcnow()
            await asyncio.sleep(random.uniform(1, 3))
            if last_other_bot_message_time and last_other_bot_message_time > start_time:
                return
            persona_used = None
            reply_from_llm = False
            if bullying and not await is_do_not_mock(message.author.id):
                reply = BULLYING_RESPONSE
            elif social_scores.get("flirtation", 0) > 0.5:
                reply = random.choice(PERSONA_REPLIES["playful"])
                persona_used = "playful"
            elif social_scores.get("avoidance", 0) > 0.5:
                reply = AVOIDANCE_REPLY
            else:
                reply = None
                use_minimal = False
                persona_override = None
                if dominant_emotion:
                    mode, persona_override = EMOTION_REPLY_MAP.get(dominant_emotion, (None, None))
                    if mode == "minimal":
                        use_minimal = True
                    elif mode == "persona":
                        persona_used = persona_override or "snarky"
                        reply = random.choice(PERSONA_REPLIES.get(persona_used, PERSONA_REPLIES["snarky"]))
                if reply is None:
                    trust = await trust_service.get_trust(message.author.id)
                    if trust < MINIMAL_REPLY_THRESHOLD or random.random() < MINIMAL_REPLY_PROB:
                        use_minimal = True
                if use_minimal:
                    reply = random.choice(MINIMAL_REPLIES)
                else:
                    persona_desc = ""
                    try:
                        persona_desc = await self.persona_manager.get_description(message.author.id)
                    except Exception:
                        logger.exception("Persona description lookup failed")
                    prompt = _build_persona_prompt([message.content], persona_desc, memory_snippets)
                    llm_reply = await generate_llm_reply(prompt)
                    if llm_reply:
                        reply = llm_reply
                        reply_from_llm = True
                    else:
                        persona = await self.persona_manager.get_persona(message.author.id)
                        persona_used = persona
                        reply = random.choice(PERSONA_REPLIES.get(persona, PERSONA_REPLIES["snarky"]))
            if reply and memory_hint and not reply_from_llm:
                reply = _append_memory_hint(reply, memory_hint)
            if FOE_MODE_CAP_ENABLED and persona_used == "snarky":
                severity, matches = evaluate_toxicity(reply)
                if severity > FOE_MODE_MAX_SEVERITY:
                    logger.info(
                        "Foe-mode cap applied: severity=%s matches=%s",
                        round(severity, 3),
                        matches,
                    )
                    reply = random.choice(MINIMAL_REPLIES)
            now = discord.utils.utcnow()
            while our_message_times and (now - our_message_times[0]).total_seconds() > BOT_MESSAGE_INTERVAL_SECONDS:
                our_message_times.popleft()
            if len(our_message_times) >= MAX_BOT_MESSAGES_PER_INTERVAL:
                return
            safe_reply = self.sanitize_response(reply)
            if safe_reply is None:
                logger.info("Reply blocked by response filter")
                return
            async def dispatch_reply() -> None:
                await self.send_filtered(message.channel, safe_reply)
                our_message_times.append(discord.utils.utcnow())
                if message.author.bot:
                    global last_bot_reply_time
                    last_bot_reply_time = discord.utils.utcnow()

            await self.enqueue_response(message.channel, dispatch_reply)

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

        for theory, conf in evaluate_triggers(message):
            await store_theory(message.author.id, theory, conf)

        await queue_deep_reflection(
            message.author.id,
            {"channel_id": message.channel.id, "message_id": message.id},
            message.content,
        )

        if hasattr(self, "process_commands") and self.user is not None:
            await self.process_commands(message)

    async def _handle_chat_raw(self, msg: Msg) -> None:
        """Send CHAT_RAW text to the monitored channel."""
        text = msg.data.decode()
        channel = self.get_channel(self.monitor_channel_id)
        try:
            safe_text = self.sanitize_response(text)
            if safe_text is None:
                logger.info("CHAT_RAW message blocked by response filter")
                if hasattr(msg, "ack") and callable(msg.ack):
                    await msg.ack()
                return
            if channel is not None:
                await self.enqueue_response(channel, lambda: self.send_filtered(channel, safe_text))
            if hasattr(msg, "ack") and callable(msg.ack):
                await msg.ack()
        except Exception:  # pragma: no cover - defensive
            logger.error("Failed to handle CHAT_RAW", exc_info=True)
            if hasattr(msg, "nak") and callable(msg.nak):
                await msg.nak()

    async def _handle_bdi_intention(self, msg: Msg) -> None:
        """React to BDI_INTENTION events by logging them."""
        try:
            data = json.loads(msg.data.decode())
            payload = BDIIntentionPayload.from_dict(data)
            logger.info("Received intention: %s", payload.goal)
            if hasattr(msg, "ack") and callable(msg.ack):
                await msg.ack()
        except Exception:  # pragma: no cover - defensive
            logger.error("Failed to handle BDI_INTENTION", exc_info=True)
            if hasattr(msg, "nak") and callable(msg.nak):
                await msg.nak()

    async def close(self) -> None:
        """Cancel background tasks and close external connections."""
        for task in self._bg_tasks:
            task.cancel()
        await asyncio.gather(*self._bg_tasks, return_exceptions=True)
        self._bg_tasks.clear()
        if self.scheduler_service is not None:
            await self.scheduler_service.stop()
            self.scheduler_service = None
        if self._subscriber is not None:
            await self._subscriber.unsubscribe_all()
            self._subscriber = None
        await db_manager.close()
        await self.response_queue.stop()
        global _nats_client, _js_context, _input_publisher, _subscriber
        if _nats_client is not None and not _nats_client.is_closed:
            await _nats_client.close()
        _nats_client = None
        _js_context = None
        _input_publisher = None
        _subscriber = None
        await super().close()


async def run(
    token: str,
    monitor_channel_id: int,
    *,
    forum_channel_id: int | None = None,
    index_channel_id: int | None = None,
    require_scheduled_events: bool = False,
    holiday_locale: str = DEFAULT_HOLIDAY_LOCALE,
) -> None:
    """Run the SocialGraphBot."""
    bot = SocialGraphBot(
        monitor_channel_id=monitor_channel_id,
        forum_channel_id=forum_channel_id,
        index_channel_id=index_channel_id,
        require_scheduled_events=require_scheduled_events,
        holiday_locale=holiday_locale,
    )
    try:
        await bot.start(token)
    finally:
        await bot.close()


async def enqueue_goal(goal: str, priority: int = 1) -> None:
    """Persist ``goal`` for later processing."""
    await init_db()
    await db_manager.add_intention(goal, priority)


if __name__ == "__main__":
    import argparse

    from deepthought.config import load_bot_env

    parser = argparse.ArgumentParser(description="Run the SocialGraphBot")
    parser.add_argument("--enqueue-goal", help="goal text to queue")
    parser.add_argument("--priority", type=int, default=1, help="goal priority")
    args = parser.parse_args()

    if args.enqueue_goal:
        asyncio.run(enqueue_goal(args.enqueue_goal, args.priority))
    else:
        env = load_bot_env()
        asyncio.run(
            run(
                env.DISCORD_TOKEN,
                env.MONITOR_CHANNEL,
                forum_channel_id=env.PROJECT_FORUM_CHANNEL_ID,
                index_channel_id=env.PROJECT_INDEX_CHANNEL_ID,
                require_scheduled_events=env.PROJECT_REQUIRE_EVENTS,
                holiday_locale=env.PROJECT_HOLIDAY_LOCALE,
            )
        )
