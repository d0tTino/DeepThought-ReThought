import asyncio
import random
import logging
import aiohttp
import aiosqlite
import discord
from textblob import TextBlob

DB_PATH = 'social_graph.db'

logger = logging.getLogger(__name__)

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
        await db.commit()

async def log_interaction(user_id: int, target_id: int) -> None:
    """Insert a user interaction record into the database."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO interactions (user_id, target_id) VALUES (?, ?)",
            (str(user_id), str(target_id)),
        )
        await db.commit()

async def send_to_prism(data: dict) -> None:
    """Send collected data to a Prism endpoint."""
    async with aiohttp.ClientSession() as session:
        await session.post("http://localhost:5000/receive_data", json=data)

def categorize_topic(text: str) -> str:
    """Very simple topic categorization based on keywords."""
    lowered = text.lower()
    if any(word in lowered for word in ("lol", "haha", "joke")):
        return "humor"
    if any(word in lowered for word in ("sad", "angry", "cry")):
        return "drama"
    return "general"

def analyze_sentiment(text: str) -> float:
    """Return sentiment polarity using TextBlob."""
    return TextBlob(text).sentiment.polarity

async def save_memory(user_id: int, topic: str, memory: str, sentiment_score: float) -> None:
    """Persist a memory entry in the database."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO memories (user_id, topic, memory, sentiment_score) VALUES (?, ?, ?, ?)",
            (str(user_id), topic, memory, sentiment_score),
        )
        await db.commit()
    logger.info(f"Stored memory for user {user_id} topic '{topic}' score {sentiment_score:.2f}")

async def recall_user(user_id: int) -> list[str]:
    """Return the latest 5 memory snippets for a user."""
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT memory FROM memories WHERE user_id=? ORDER BY timestamp DESC LIMIT 5",
            (str(user_id),),
        ) as cursor:
            rows = await cursor.fetchall()
    logger.info(f"Recalling {len(rows)} memories for user {user_id}")
    return [row[0] for row in rows]

async def monitor_channels(bot: discord.Client, channel_id: int) -> None:
    """Periodically monitor a channel and prompt activity if quiet."""
    await bot.wait_until_ready()
    channel = bot.get_channel(channel_id)
    while not bot.is_closed():
        # Fetch the last 10 messages
        messages = [m async for m in channel.history(limit=10)]
        if not messages:
            await channel.send("The channel is quiet... Let's start a conversation!")
        await asyncio.sleep(300)  # check every 5 minutes

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

    async def on_message(self, message: discord.Message) -> None:
        if message.author == self.user:
            return

        # Log the interaction
        await log_interaction(message.author.id, message.channel.id)

        # Analyze sentiment and potentially store memory
        sentiment = analyze_sentiment(message.content)
        topic = categorize_topic(message.content)
        if abs(sentiment) > 0.5:
            await save_memory(message.author.id, topic, message.content, sentiment)

        await asyncio.sleep(random.uniform(1, 3))  # simulate thinking
        await message.channel.send("I'm pondering your message...")

        await send_to_prism({
            "user_id": str(message.author.id),
            "channel_id": str(message.channel.id),
            "content": message.content,
        })

        memories = await recall_user(message.author.id)
        if memories:
            logger.info(f"Recalling memories for {message.author.id}: {memories}")


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
