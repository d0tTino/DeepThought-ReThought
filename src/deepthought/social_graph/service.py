from __future__ import annotations

import asyncio
import json
import logging
import random
from datetime import timezone
from typing import Tuple

import discord

from . import db_manager as global_db
from .constants import (BOT_CHAT_ENABLED, IDLE_TIMEOUT_MINUTES,
                        PLAYFUL_REPLY_TIMEOUT_MINUTES,
                        REFLECTION_CHECK_SECONDS, idle_response_candidates)
from .db_manager import DBManager
from .utils import analyze_sentiment

logger = logging.getLogger(__name__)


class SocialGraphService:
    """Background tasks for monitoring channels and processing reflections."""

    def __init__(self, db_manager: DBManager | None = None) -> None:
        self.db = db_manager or global_db.db_manager

    async def assign_themes(self) -> None:
        rows = await self.db.get_all_sentiment_trends()
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
            await self.db.set_theme(user_id, channel_id, theme)

    def generate_reflection(self, prompt: str) -> str:
        polarity = analyze_sentiment(prompt)
        if polarity > 0.1:
            mood = "positive"
        elif polarity < -0.1:
            mood = "negative"
        else:
            mood = "neutral"
        return f"Your message felt {mood}."

    async def process_deep_reflections(self, bot: discord.Client) -> None:
        await bot.wait_until_ready()
        while not bot.is_closed():
            try:
                rows = await self.db.list_pending_tasks()
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
                        reflection = self.generate_reflection(prompt)
                        logger.info("Posting deep reflection for task %s", task_id)
                        await channel.send(
                            f"After some thought... {reflection}",
                            reference=ref,
                        )
                    await self.db.mark_task_done(task_id)
                await self.assign_themes()
                await asyncio.sleep(REFLECTION_CHECK_SECONDS)
            except asyncio.CancelledError:
                logger.info("process_deep_reflections cancelled")
                break

    async def who_is_active(
        self, channel: discord.TextChannel, limit: int = 20
    ) -> Tuple[set, set]:
        bots = set()
        humans = set()
        async for msg in channel.history(limit=limit):
            if msg.author.bot:
                bots.add(msg.author.id)
            else:
                humans.add(msg.author.id)
        return bots, humans

    async def last_human_message_age(
        self, channel: discord.TextChannel, limit: int = 50
    ):
        async for msg in channel.history(limit=limit):
            if not msg.author.bot:
                return (
                    discord.utils.utcnow() - msg.created_at.replace(tzinfo=timezone.utc)
                ).total_seconds() / 60
        return None

    async def monitor_channels(self, bot: discord.Client, channel_id: int) -> None:
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
                # fmt: off
                if last_message and last_message.author.bot and prev_message and not prev_message.author.bot:
                    age = (
                        discord.utils.utcnow() - prev_message.created_at.replace(tzinfo=timezone.utc)
                    ).total_seconds() / 60
                    if age < PLAYFUL_REPLY_TIMEOUT_MINUTES:
                        await asyncio.sleep(60)
                        continue
                # fmt: on

                if not last_message:
                    send_prompt = True
                else:
                    # fmt: off
                    idle_minutes = (
                        discord.utils.utcnow() - last_message.created_at.replace(tzinfo=timezone.utc)
                    ).total_seconds() / 60
                    # fmt: on
                    if idle_minutes >= IDLE_TIMEOUT_MINUTES:
                        send_prompt = True
                    elif BOT_CHAT_ENABLED:
                        bots, humans = await self.who_is_active(channel)
                        if bots and not humans:
                            age = await self.last_human_message_age(channel)
                            if age is None or age >= PLAYFUL_REPLY_TIMEOUT_MINUTES:
                                send_prompt = True
                                if last_message.author.bot:
                                    respond_to = last_message

                if send_prompt:
                    from . import generate_idle_response

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


__all__ = ["SocialGraphService"]
