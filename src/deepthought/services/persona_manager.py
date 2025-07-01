from __future__ import annotations

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

    async def get_persona(self, user_id: int) -> str:
        await self._db.init_db()
        affinity = await self._db.get_affinity(user_id)
        if affinity >= self._friendly:
            return "friendly"
        if affinity >= self._playful:
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
