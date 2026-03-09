from __future__ import annotations

from nats.aio.client import Client as NATS
from nats.js.client import JetStreamContext

from deepthought.config import load_discord_bot_token
from deepthought.eda import Publisher, Subscriber
from deepthought.motivate import Ledger, RewardManager


class TemplateService:
    """Container service running :class:`RewardManager`."""

    def __init__(self, nats_client: NATS, js_context: JetStreamContext) -> None:
        self._publisher = Publisher(nats_client, js_context)
        self._subscriber = Subscriber(nats_client, js_context)
        self._ledger = Ledger(nats_client, js_context)
        token = load_discord_bot_token()
        self._manager = RewardManager(
            self._subscriber, self._ledger, self._publisher, token
        )

    async def start(self, durable_name: str = "reward_listener") -> bool:
        return await self._manager.start_listening(durable_name)

    async def stop(self) -> None:
        await self._manager.stop_listening()
