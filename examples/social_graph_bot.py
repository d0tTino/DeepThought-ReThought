import asyncio
import os
import random
from datetime import timezone

import aiohttp
import aiosqlite
import discord

DB_PATH = 'social_graph.db'

# Configuration values
MAX_BOT_SPEAKERS = int(os.getenv("MAX_BOT_SPEAKERS", "2"))
IDLE_TIMEOUT_MINUTES = int(os.getenv("IDLE_TIMEOUT_MINUTES", "5"))

# Candidate prompts used when the bot speaks after a period of silence
idle_response_candidates = [
    "Ever feel like everyone vanished?",
    "I'm still here if anyone wants to chat!",
    "Silence can be golden, but conversation is better.",
]

async def init_db():
    """Initialize the SQLite database for tracking interactions."""
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
            idle_minutes = (discord.utils.utcnow() - last_message.created_at.replace(tzinfo=timezone.utc)).total_seconds() / 60
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

    async def on_message(self, message: discord.Message) -> None:
        if message.author == self.user:
            return

        bots, _ = await who_is_active(message.channel)
        if len(bots) > MAX_BOT_SPEAKERS and self.user not in message.mentions:
            # Too many bots talking and we're not addressed directly
            return

        # Log the interaction
        await log_interaction(message.author.id, message.channel.id)

        async with message.channel.typing():
            await asyncio.sleep(random.uniform(1, 3))
            await message.channel.send("I'm pondering your message...")

        await send_to_prism({
            "user_id": str(message.author.id),
            "channel_id": str(message.channel.id),
            "content": message.content,
        })


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
