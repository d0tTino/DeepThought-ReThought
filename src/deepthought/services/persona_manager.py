from __future__ import annotations

import asyncio
import random

from .db_manager import DBManager


class PersonaManager:
    """Select prompts based on user affinity and pairwise relationships.

    When a ``target_id`` is unavailable, persona selection falls back to
    per-user affinity/trust scores instead of pairwise relationship data.
    """

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
        self._personality: dict[str, dict[str, float]] = {}

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

    def update_personality(self, user_id: int, traits: dict[str, float]) -> None:
        """Update stored personality ``traits`` for ``user_id``.

        The provided ``traits`` mapping is merged with any existing traits for
        the user, overwriting values for matching keys.
        """

        uid = str(user_id)
        current = self._personality.setdefault(uid, {})
        current.update(traits)

    async def get_persona(self, user_id: int | str, target_id: int | str | None = None) -> str:
        if self._init_task is not None:
            await self._init_task
            self._init_task = None
        traits = self._personality.get(str(user_id), {})
        if target_id is None:
            score = await self._db.get_mutual_affinity(user_id)
        else:
            relationship = await self._db.get_relationship_type(user_id, target_id)
            if relationship == "friend":
                return "friendly"
            if relationship == "rival":
                return "snarky"
            score = await self._db.get_pair_mutual_affinity(user_id, target_id)
        if score + traits.get("friendly", 0) >= self._friendly:
            return "friendly"
        if score + traits.get("playful", 0) >= self._playful:
            return "playful"
        return "snarky"

    async def choose_prompt(
        self, user_id: int | str, prompts: dict[str, list[str]], target_id: int | str | None = None
    ) -> str:
        persona = await self.get_persona(user_id, target_id)
        options = prompts.get(persona) or prompts.get("default") or []
        if not options:
            return ""
        return random.choice(options)

    async def get_description(self, user_id: int | str, target_id: int | str | None = None) -> str:
        """Return a persona description for ``user_id`` if available."""
        persona = await self.get_persona(user_id, target_id)
        return self._descriptions.get(persona, "")
