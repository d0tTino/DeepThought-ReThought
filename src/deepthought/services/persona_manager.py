from __future__ import annotations

import random

from examples import social_graph_bot as sg


class PersonaManager:
    """Select prompts based on user affinity."""

    def __init__(self, db_manager: sg.DBManager, friendly: int = 5, playful: int = 2) -> None:
        self._db = db_manager
        self._friendly = friendly
        self._playful = playful

    async def get_persona(self, user_id: int) -> str:
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
