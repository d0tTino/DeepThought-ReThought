"""RewardManager subscribes to chat events and scores bot output."""

from __future__ import annotations

import json
import logging
from collections import deque
from typing import Deque, Optional

import aiohttp
import numpy as np
from nats.aio.msg import Msg
from sentence_transformers import SentenceTransformer, util

from ..config import get_settings
from ..eda.publisher import Publisher
from ..eda.subscriber import Subscriber
from .ledger import Ledger

logger = logging.getLogger(__name__)


class RewardManager:
    """Subscribe to ``chat.bot`` messages and compute reward scores."""

    def __init__(
        self,
        subscriber: Subscriber,
        ledger: Ledger,
        publisher: Publisher,
        discord_token: str,
        model: SentenceTransformer | None = None,
    ) -> None:
        self._subscriber = subscriber
        self._ledger = ledger
        self._publisher = publisher
        self._token = discord_token

        settings = get_settings().reward
        self._novelty_threshold = settings.novelty_threshold
        self._social_threshold = settings.social_affinity_threshold
        self._novelty_weight = settings.novelty_weight
        self._social_weight = settings.social_weight
        self._window: Deque[np.ndarray] = deque(maxlen=settings.window_size)

        self._model = model or SentenceTransformer("all-MiniLM-L6-v2")

    async def start_listening(self, durable_name: str = "reward_listener") -> bool:
        """Begin consuming ``chat.bot`` messages."""
        try:
            await self._subscriber.subscribe(
                subject="chat.bot",
                handler=self._handle_chat_event,
                use_jetstream=True,
                durable=durable_name,
            )
            return True
        except Exception as exc:  # pragma: no cover - network errors
            logger.error("RewardManager failed to subscribe: %s", exc, exc_info=True)
            return False

    async def stop_listening(self) -> None:
        """Unsubscribe from all subjects."""
        await self._subscriber.unsubscribe_all()

    async def _handle_chat_event(self, msg: Msg) -> None:
        """Process a bot message and publish the reward."""
        prompt = ""
        response = ""
        channel_id = None
        message_id = None
        try:
            data = json.loads(msg.data.decode())
            if not isinstance(data, dict):
                raise ValueError("payload must be a dict")
            prompt = str(data.get("prompt", ""))
            response = str(data.get("response", data.get("content", "")))
            channel_id = data.get("channel_id")
            message_id = data.get("message_id")
        except Exception as exc:  # pragma: no cover - bad payload
            logger.error("Invalid chat.bot payload: %s", exc)
            await msg.ack()
            return

        novelty = self._score_novelty(response)
        social = await self._score_social(channel_id, message_id)
        reward = (
            float(novelty >= self._novelty_threshold) * self._novelty_weight
            + float(social >= self._social_threshold) * self._social_weight  # noqa: W503
        )

        try:
            await self._ledger.publish(prompt, response, reward)
            await self._publisher.publish("agent.reward", {"reward": reward}, use_jetstream=True)
            await msg.ack()
        except Exception as exc:  # pragma: no cover - publish failure
            logger.error("Failed to publish reward: %s", exc, exc_info=True)

    def _score_novelty(self, text: str) -> float:
        """Return novelty score based on cosine distance to previous messages."""
        emb = self._model.encode(text, convert_to_numpy=True)
        if not self._window:
            self._window.append(emb)
            return 1.0
        sims = util.cos_sim(emb, np.stack(list(self._window))).flatten().tolist()
        max_sim = max(sims) if sims else 0.0
        self._window.append(emb)
        return 1.0 - float(max_sim)

    async def _score_social(self, channel_id: Optional[int], message_id: Optional[int]) -> int:
        """Return the total reaction count for the Discord message."""
        if channel_id is None or message_id is None:
            return 0
        url = f"https://discord.com/api/v10/channels/{channel_id}/messages/{message_id}"
        headers = {"Authorization": f"Bot {self._token}"}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as resp:
                    if resp.status != 200:
                        return 0
                    payload = await resp.json()
                    reactions = payload.get("reactions", [])
                    return int(sum(r.get("count", 0) for r in reactions))
        except Exception as exc:  # pragma: no cover - network issues
            logger.warning("Failed to fetch reactions: %s", exc)
            return 0
