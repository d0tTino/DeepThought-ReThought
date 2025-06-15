import asyncio
import json
import logging
import os
import random
from datetime import timezone
from typing import List, Tuple

import aiohttp
import aiosqlite
import discord
from textblob import TextBlob

logger = logging.getLogger(__name__)

DB_PATH = "social_graph.db"

# Configuration values
MAX_BOT_SPEAKERS = int(os.getenv("MAX_BOT_SPEAKERS", "2"))
IDLE_TIMEOUT_MINUTES = int(os.getenv("IDLE_TIMEOUT_MINUTES", "5"))
REFLECTION_CHECK_SECONDS = int(os.getenv("REFLECTION_CHECK_SECONDS", "300"))

# Candidate prompts used when the bot speaks after a period of silence
idle_response_candidates = [
    "Ever feel like everyone vanished?",
    "I'm still here if anyone wants to chat!",
    "Silence can be golden, but conversation is better.",
]


async def init_db():
    """Initialize the SQLite database for tracking interactions and memories."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS interactions (
                user_id TEXT,
                target_id TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        await db.execute(
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
        await db.execute(
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
        await db.execute(
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
        await db.commit()


async def log_interaction(user_id: int, target_id: int) -> None:
    """Insert a user interaction record into the database."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO interactions (user_id, target_id) VALUES (?, ?)",
            (str(user_id), str(target_id)),
        )
        await db.commit()


async def recall_user(user_id: int):
    """Retrieve memories for a given user."""
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT topic, memory FROM memories WHERE user_id = ?", (str(user_id),)) as cur:
            return await cur.fetchall()


async def store_memory(user_id: int, text: str, score: float, topic: str = "message") -> None:
    """Persist the raw message text and sentiment score."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO memories (user_id, topic, memory, sentiment_score) VALUES (?, ?, ?, ?)",
            (str(user_id), topic, text, score),
        )
        await db.commit()


async def send_to_prism(data: dict) -> None:
    """Send collected data to a Prism endpoint."""
    try:
        async with aiohttp.ClientSession() as session:
            await session.post("http://localhost:5000/receive_data", json=data, timeout=5)
    except Exception as exc:
        logger.warning("Failed to send data to Prism: %s", exc)


async def store_theory(subject_id: int, theory: str, confidence: float) -> None:
    """Persist an inferred theory about a user."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO theories (subject_id, theory, confidence)
            VALUES (?, ?, ?)
            ON CONFLICT(subject_id, theory) DO UPDATE SET
                confidence=excluded.confidence,
                updated=CURRENT_TIMESTAMP
            """,
            (str(subject_id), theory, confidence),
        )
        await db.commit()


async def get_theories(subject_id: int):
    """Retrieve stored theories about a subject."""
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT theory, confidence FROM theories WHERE subject_id=?",
            (str(subject_id),),
        ) as cur:
            return await cur.fetchall()


async def queue_deep_reflection(user_id: int, context: dict, prompt: str) -> int:
    """Add a deep reflection task to the queue."""
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "INSERT INTO queued_tasks (user_id, context, prompt) VALUES (?, ?, ?)",
            (str(user_id), json.dumps(context), prompt),
        )
        await db.commit()
        return cur.lastrowid


def generate_reflection(prompt: str) -> str:
    """Return a simple reflection string based on sentiment analysis."""
    blob = TextBlob(prompt)
    polarity = blob.sentiment.polarity
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
        async with aiosqlite.connect(DB_PATH) as db:
            async with db.execute(
                "SELECT task_id, user_id, context, prompt FROM queued_tasks WHERE status='pending'"
            ) as cur:
                rows = await cur.fetchall()
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
                await db.execute(
                    "UPDATE queued_tasks SET status='done' WHERE task_id=?",
                    (task_id,),
                )
            await db.commit()
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


async def monitor_channels(bot: discord.Client, channel_id: int) -> None:
    """Monitor a channel and occasionally speak during idle periods."""
    await bot.wait_until_ready()
    channel = bot.get_channel(channel_id)
    while not bot.is_closed():
        last_message = None
        async for msg in channel.history(limit=1):
            last_message = msg
            break

        if not last_message:
            prompt = random.choice(idle_response_candidates)
            async with channel.typing():
                await asyncio.sleep(random.uniform(3, 10))
                await channel.send(prompt)
        else:
            idle_minutes = (
                discord.utils.utcnow() - last_message.created_at.replace(tzinfo=timezone.utc)
            ).total_seconds() / 60
            if idle_minutes >= IDLE_TIMEOUT_MINUTES:
                prompt = random.choice(idle_response_candidates)
                async with channel.typing():
                    await asyncio.sleep(random.uniform(3, 10))
                    await channel.send(prompt)
        await asyncio.sleep(60)


class SocialGraphBot(discord.Client):
    """Discord bot that records interactions and demonstrates simple awareness."""

    def __init__(self, *args, monitor_channel_id: int, **kwargs):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(*args, intents=intents, **kwargs)
        self.monitor_channel_id = monitor_channel_id

    async def setup_hook(self) -> None:
        await init_db()
        self.loop.create_task(monitor_channels(self, self.monitor_channel_id))
        self.loop.create_task(process_deep_reflections(self))

    async def on_message(self, message: discord.Message) -> None:
        if message.author == self.user:
            return

        blob = TextBlob(message.content)
        score = blob.sentiment.polarity
        await store_memory(message.author.id, message.content, score)

        bots, _ = await who_is_active(message.channel)
        if len(bots) > MAX_BOT_SPEAKERS and self.user not in message.mentions:
            # Too many bots talking and we're not addressed directly
            return

        # Log the interaction
        await log_interaction(message.author.id, message.channel.id)

        async with message.channel.typing():
            await asyncio.sleep(random.uniform(1, 3))
            await message.channel.send("I'm pondering your message...")

        await send_to_prism(
            {
                "user_id": str(message.author.id),
                "channel_id": str(message.channel.id),
                "content": message.content,
            }
        )

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
