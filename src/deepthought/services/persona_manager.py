from __future__ import annotations

import asyncio
import random

from .db_manager import DBManager


class PersonaManager:
    """Select prompts based on user affinity."""

    def __init__(
        self,
        db_manager: DBManager,
        friendly: int = 5,
        playful: int = 2,
        descriptions: dict[str, str] | None = None,
    ) -> None:
        self._db = db_manager
        self._friendly = friendly
        self._playful = playful
        self._descriptions = descriptions or {}

        # Initialize the database once. If an event loop is running we
        # schedule the task and await it on first use. Otherwise we
        # perform the initialization synchronously.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # no running loop
            loop = None

        if loop and loop.is_running():
            self._init_task = loop.create_task(self._db.init_db())
        else:  # pragma: no cover - typically executed outside tests
            asyncio.run(self._db.init_db())
            self._init_task = None

    async def get_persona(self, user_id: int) -> str:
        if self._init_task is not None:
            await self._init_task
            self._init_task = None
        score = await self._db.get_mutual_affinity(user_id)
        if score >= self._friendly:
            return "friendly"
        if score >= self._playful:
            return "playful"
        return "snarky"

    async def choose_prompt(self, user_id: int, prompts: dict[str, list[str]]) -> str:
        persona = await self.get_persona(user_id)
        options = prompts.get(persona) or prompts.get("default") or []
        if not options:
            return ""
        return random.choice(options)

    async def get_description(self, user_id: int) -> str:
        """Return a persona description for ``user_id`` if available."""
        persona = await self.get_persona(user_id)
        return self._descriptions.get(persona, "")
