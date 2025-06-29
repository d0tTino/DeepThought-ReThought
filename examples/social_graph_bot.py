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
import discord
import nats
from nats.aio.client import Client as NATS
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


from deepthought.config import get_settings
from deepthought.eda.events import EventSubjects, InputReceivedPayload
from deepthought.eda.publisher import Publisher

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

DB_PATH = os.getenv("SOCIAL_GRAPH_DB", "social_graph.db")
CURRENT_DB_PATH = DB_PATH


# Endpoint for forwarding collected data
PRISM_ENDPOINT = os.getenv("PRISM_ENDPOINT", "http://localhost:5000/receive_data")

# NATS configuration for publishing events
NATS_URL = os.getenv("NATS_URL", "nats://localhost:4222")
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
        gen_prompt = prompt or os.getenv(
            "IDLE_GENERATOR_PROMPT", "Say something to spark conversation."
        )
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

# Limits used when validating inputs
MAX_MEMORY_LENGTH = 1000
MAX_THEORY_LENGTH = 256
MAX_PROMPT_LENGTH = 2000


class DBManager:
    """Lightweight wrapper managing a single aiosqlite connection."""

    def __init__(self, db_path: str = DB_PATH) -> None:
        self.db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        if self._db is None:
            dir_path = os.path.dirname(self.db_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            self._db = await aiosqlite.connect(self.db_path)

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def init_db(self) -> None:
        await self.connect()
        assert self._db
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS interactions (
                user_id TEXT,
                target_id TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                user_id TEXT,
                topic TEXT,
                memory TEXT,
                sentiment_score REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS theories (
                subject_id TEXT,
                theory TEXT,
                confidence REAL,
                updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(subject_id, theory)
            )
            """
        )
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS queued_tasks (
                task_id INTEGER PRIMARY KEY,
                user_id TEXT,
                context TEXT,
                prompt TEXT,
                status TEXT DEFAULT 'pending',
                created TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS sentiment_trends (
                user_id TEXT,
                channel_id TEXT,
                sentiment_sum REAL DEFAULT 0,
                message_count INTEGER DEFAULT 0,
                PRIMARY KEY(user_id, channel_id)

            )
            """
        )
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS themes (
                user_id TEXT,
                channel_id TEXT,
                theme TEXT,
                updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(user_id, channel_id)
            )
            """
        )
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS user_flags (
                user_id TEXT PRIMARY KEY,
                do_not_mock INTEGER
            )
            """
        )
        await self._db.commit()

    async def log_interaction(self, user_id: int, target_id: int) -> None:
        await self.connect()
        assert self._db
        await self._db.execute(
            "INSERT INTO interactions (user_id, target_id) VALUES (?, ?)",
            (str(user_id), str(target_id)),
        )
        await self._db.commit()

    async def recall_user(self, user_id: int):
        await self.connect()
        assert self._db
        async with self._db.execute(
            "SELECT topic, memory FROM memories WHERE user_id= ?",
            (str(user_id),),
        ) as cur:
            return await cur.fetchall()

    async def store_memory(
        self,
        user_id: int,
        memory: str,
        topic: str = "",
        sentiment_score: float | None = None,
    ) -> None:
        if not isinstance(memory, str) or not memory.strip():
            raise ValueError("memory must be a non-empty string")
        if len(memory) > MAX_MEMORY_LENGTH:
            raise ValueError("memory exceeds maximum length")
        if not isinstance(topic, str):
            raise ValueError("topic must be a string")
        if sentiment_score is not None:
            if not isinstance(sentiment_score, (int, float)):
                raise ValueError("sentiment_score must be numeric")
            if not -1 <= float(sentiment_score) <= 1:
                raise ValueError("sentiment_score out of range")

        await self.connect()
        assert self._db
        await self._db.execute(
            "INSERT INTO memories (user_id, topic, memory, sentiment_score) VALUES (?, ?, ?, ?)",
            (str(user_id), topic, memory, sentiment_score),
        )
        await self._db.commit()

    async def store_theory(
        self, subject_id: int, theory: str, confidence: float
    ) -> None:
        if not isinstance(theory, str) or not theory.strip():
            raise ValueError("theory must be a non-empty string")
        if len(theory) > MAX_THEORY_LENGTH:
            raise ValueError("theory exceeds maximum length")
        if not isinstance(confidence, (int, float)):
            raise ValueError("confidence must be numeric")
        if not 0 <= float(confidence) <= 1:
            raise ValueError("confidence out of range")

        await self.connect()
        assert self._db
        await self._db.execute(
            """
            INSERT INTO theories (subject_id, theory, confidence)
            VALUES (?, ?, ?)
            ON CONFLICT(subject_id, theory) DO UPDATE SET
                confidence=excluded.confidence,
                updated=CURRENT_TIMESTAMP
            """,
            (str(subject_id), theory, confidence),
        )
        await self._db.commit()

    async def get_theories(self, subject_id: int):
        await self.connect()
        assert self._db
        async with self._db.execute(
            "SELECT theory, confidence FROM theories WHERE subject_id=?",
            (str(subject_id),),
        ) as cur:
            return await cur.fetchall()

    async def update_sentiment_trend(
        self,
        user_id: int,
        channel_id: int,
        sentiment_score: float,
    ) -> None:
        if not isinstance(sentiment_score, (int, float)):
            raise ValueError("sentiment_score must be numeric")
        if not -1 <= float(sentiment_score) <= 1:
            raise ValueError("sentiment_score out of range")
        await self.connect()
        assert self._db
        await self._db.execute(
            """
            INSERT INTO sentiment_trends (user_id, channel_id, sentiment_sum, message_count)
            VALUES (?, ?, ?, 1)
            ON CONFLICT(user_id, channel_id) DO UPDATE SET
                sentiment_sum=sentiment_trends.sentiment_sum + excluded.sentiment_sum,
                message_count=sentiment_trends.message_count + 1
            """,
            (str(user_id), str(channel_id), sentiment_score),
        )
        await self._db.commit()

    async def get_sentiment_trend(self, user_id: int, channel_id: int):
        await self.connect()
        assert self._db
        async with self._db.execute(
            "SELECT sentiment_sum, message_count FROM sentiment_trends WHERE user_id=? AND channel_id=?",
            (str(user_id), str(channel_id)),
        ) as cur:
            return await cur.fetchone()

    async def queue_deep_reflection(
        self, user_id: int, context: dict, prompt: str
    ) -> int:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if len(prompt) > MAX_PROMPT_LENGTH:
            raise ValueError("prompt exceeds maximum length")
        if not isinstance(context, dict):
            raise ValueError("context must be a dictionary")
        try:
            context_json = json.dumps(context)
        except (TypeError, ValueError) as exc:
            raise ValueError("context is not JSON serializable") from exc

        await self.connect()
        assert self._db
        cur = await self._db.execute(
            "INSERT INTO queued_tasks (user_id, context, prompt) VALUES (?, ?, ?)",
            (str(user_id), context_json, prompt),
        )
        await self._db.commit()
        return cur.lastrowid

    async def list_pending_tasks(self):
        """Return pending reflection tasks."""
        await self.connect()
        assert self._db
        async with self._db.execute(
            "SELECT task_id, user_id, context, prompt FROM queued_tasks WHERE status='pending'"
        ) as cur:
            return await cur.fetchall()

    async def mark_task_done(self, task_id: int) -> None:
        """Mark a queued task as completed."""
        await self.connect()
        assert self._db
        await self._db.execute(
            "UPDATE queued_tasks SET status='done' WHERE task_id=?",
            (task_id,),
        )
        await self._db.commit()

    async def set_do_not_mock(self, user_id: int, flag: bool = True) -> None:
        await self.connect()
        assert self._db
        await self._db.execute(
            """
            INSERT INTO user_flags (user_id, do_not_mock)
            VALUES (?, ?)
            ON CONFLICT(user_id) DO UPDATE SET do_not_mock=excluded.do_not_mock
            """,
            (str(user_id), int(flag)),
        )
        await self._db.commit()

    async def is_do_not_mock(self, user_id: int) -> bool:
        await self.connect()
        assert self._db
        async with self._db.execute(
            "SELECT do_not_mock FROM user_flags WHERE user_id=?",
            (str(user_id),),
        ) as cur:
            row = await cur.fetchone()
            return bool(row[0]) if row else False

    async def set_theme(self, user_id: int, channel_id: int, theme: str) -> None:
        if not isinstance(theme, str) or not theme.strip():
            raise ValueError("theme must be a non-empty string")
        await self.connect()
        assert self._db
        await self._db.execute(
            """
            INSERT INTO themes (user_id, channel_id, theme)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id, channel_id) DO UPDATE SET
                theme=excluded.theme,
                updated=CURRENT_TIMESTAMP
            """,
            (str(user_id), str(channel_id), theme),
        )
        await self._db.commit()

    async def get_theme(self, user_id: int, channel_id: int):
        await self.connect()
        assert self._db
        async with self._db.execute(
            "SELECT theme FROM themes WHERE user_id=? AND channel_id=?",
            (str(user_id), str(channel_id)),
        ) as cur:
            row = await cur.fetchone()
            return row[0] if row else None

    async def get_all_sentiment_trends(self):
        await self.connect()
        assert self._db
        async with self._db.execute(
            "SELECT user_id, channel_id, sentiment_sum, message_count FROM sentiment_trends"
        ) as cur:
            return await cur.fetchall()


DEFAULT_DB_PATH = DB_PATH
db_manager = DBManager()


async def init_db(db_path: str | None = None) -> None:
    """Initialize the database, recreating the manager when the path changes."""
    global db_manager, CURRENT_DB_PATH

    target_path = (
        db_path
        if db_path is not None
        else (
            DB_PATH
            if DB_PATH != CURRENT_DB_PATH and db_manager.db_path == CURRENT_DB_PATH
            else db_manager.db_path
        )
    )

    if db_manager.db_path != target_path:
        if db_manager._db is not None:
            await db_manager.close()
        db_manager = DBManager(target_path)

    await db_manager.init_db()
    CURRENT_DB_PATH = db_manager.db_path


async def log_interaction(user_id: int, target_id: int) -> None:
    await db_manager.log_interaction(user_id, target_id)


async def recall_user(user_id: int):
    return await db_manager.recall_user(user_id)


async def store_memory(
    user_id: int,
    memory: str,
    topic: str = "",
    sentiment_score: float | None = None,
) -> None:
    await db_manager.store_memory(
        user_id, memory, topic=topic, sentiment_score=sentiment_score
    )


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
    await _ensure_nats()
    if _input_publisher is None:
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


async def queue_deep_reflection(user_id: int, context: dict, prompt: str) -> int:
    return await db_manager.queue_deep_reflection(user_id, context, prompt)


async def set_do_not_mock(user_id: int, flag: bool = True) -> None:
    await db_manager.set_do_not_mock(user_id, flag)


async def is_do_not_mock(user_id: int) -> bool:
    return await db_manager.is_do_not_mock(user_id)


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
    """Return sets of bot and human authors from recent messages."""
    bots = set()
    humans = set()
    async for msg in channel.history(limit=limit):
        if msg.author.bot:
            bots.add(msg.author.id)
        else:
            humans.add(msg.author.id)
    return bots, humans


async def last_human_message_age(channel: discord.TextChannel, limit: int = 50):
    """Return minutes since the most recent human message or ``None`` if none."""
    async for msg in channel.history(limit=limit):
        if not msg.author.bot:
            return (
                discord.utils.utcnow() - msg.created_at.replace(tzinfo=timezone.utc)
            ).total_seconds() / 60
    return None


async def monitor_channels(bot: discord.Client, channel_id: int) -> None:
    """Monitor a channel and occasionally speak during idle periods."""
    await bot.wait_until_ready()
    channel = bot.get_channel(channel_id)
    if channel is None:
        logger.error("Channel %s does not exist", channel_id)
        return
    while not bot.is_closed():
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
        if (
            last_message
            and last_message.author.bot
            and prev_message
            and not prev_message.author.bot
        ):
            age = (
                discord.utils.utcnow()
                - prev_message.created_at.replace(tzinfo=timezone.utc)
            ).total_seconds() / 60
            if age < PLAYFUL_REPLY_TIMEOUT_MINUTES:
                await asyncio.sleep(60)
                continue

        if not last_message:
            send_prompt = True
        else:
            idle_minutes = (
                discord.utils.utcnow()
                - last_message.created_at.replace(tzinfo=timezone.utc)
            ).total_seconds() / 60
            if idle_minutes >= IDLE_TIMEOUT_MINUTES:
                send_prompt = True
            elif BOT_CHAT_ENABLED:
                bots, humans = await who_is_active(channel)
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


class SocialGraphBot(discord.Client):
    """Discord bot that records interactions and demonstrates simple awareness."""

    def __init__(self, *args, monitor_channel_id: int, **kwargs):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        intents.presences = True
        super().__init__(*args, intents=intents, **kwargs)
        self.monitor_channel_id = monitor_channel_id

    async def setup_hook(self) -> None:
        await db_manager.connect()
        await init_db()
        self.loop.create_task(monitor_channels(self, self.monitor_channel_id))
        self.loop.create_task(process_deep_reflections(self))

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

        bots, _ = await who_is_active(message.channel)
        if len(bots) > MAX_BOT_SPEAKERS and self.user not in message.mentions:
            # Too many bots talking and we're not addressed directly
            return

        # Log the interaction
        await log_interaction(message.author.id, message.channel.id)

        async with message.channel.typing():
            await asyncio.sleep(random.uniform(1, 3))
            async for recent in message.channel.history(limit=1):
                if recent.id != message.id and recent.author.bot:
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


def run(token: str, monitor_channel_id: int) -> None:
    """Run the SocialGraphBot."""
    bot = SocialGraphBot(monitor_channel_id=monitor_channel_id)
    bot.run(token)


if __name__ == "__main__":
    import os

    token = os.getenv("DISCORD_TOKEN")
    channel_id = int(os.getenv("MONITOR_CHANNEL", "0"))
    if not token or channel_id == 0:
        print("Please set DISCORD_TOKEN and MONITOR_CHANNEL environment variables.")
    else:
        run(token, channel_id)
