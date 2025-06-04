import asyncio
import random
import aiohttp
import aiosqlite
import discord

DB_PATH = 'social_graph.db'

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

        await asyncio.sleep(random.uniform(1, 3))  # simulate thinking
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
