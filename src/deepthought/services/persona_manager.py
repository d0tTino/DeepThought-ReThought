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
        *,
        friendly_exit: float | None = None,
        playful_exit: float | None = None,
        hysteresis: float = 0.0,
        sentiment_hysteresis: float = 0.0,
        sentiment_weight: float = 0.0,
        descriptions: dict[str, str] | None = None,
    ) -> None:
        self._db = db_manager
        self._friendly = friendly
        self._playful = playful
        self._friendly_exit = (
            friendly_exit if friendly_exit is not None else friendly - hysteresis
        )
        self._playful_exit = (
            playful_exit if playful_exit is not None else playful - hysteresis
        )
        self._sentiment_hysteresis = sentiment_hysteresis
        self._sentiment_weight = sentiment_weight
        self._descriptions = descriptions or {}
        self._personality: dict[str, dict[str, float]] = {}
        self._persona_state: dict[str, str] = {}
        self._relationship_state: dict[str, str | None] = {}
        self._sentiment_state: dict[str, float] = {}

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

    async def get_persona(
        self,
        user_id: int | str,
        target_id: int | str | None = None,
        *,
        channel_id: int | str | None = None,
    ) -> str:
        if self._init_task is not None:
            await self._init_task
            self._init_task = None
        traits = self._personality.get(str(user_id), {})
        relationship = None
        key = self._state_key(user_id, target_id, channel_id)
        if target_id is None:
            score = await self._db.get_mutual_affinity(user_id)
        else:
            relationship = await self._db.get_relationship_type(user_id, target_id)
            score = await self._db.get_pair_mutual_affinity(user_id, target_id)
        sentiment_adjustment, average_sentiment = await self._get_sentiment_context(
            user_id, channel_id
        )
        score += sentiment_adjustment
        persona = self._select_persona(
            score,
            traits,
            relationship,
            key,
            average_sentiment,
        )
        self._persona_state[key] = persona
        self._relationship_state[key] = relationship
        if average_sentiment is not None:
            self._sentiment_state[key] = average_sentiment
        return persona

    async def choose_prompt(
        self,
        user_id: int | str,
        prompts: dict[str, list[str]],
        target_id: int | str | None = None,
        *,
        channel_id: int | str | None = None,
    ) -> str:
        persona = await self.get_persona(user_id, target_id, channel_id=channel_id)
        options = prompts.get(persona) or prompts.get("default") or []
        if not options:
            return ""
        return random.choice(options)

    async def get_description(
        self,
        user_id: int | str,
        target_id: int | str | None = None,
        *,
        channel_id: int | str | None = None,
    ) -> str:
        """Return a persona description for ``user_id`` if available."""
        persona = await self.get_persona(user_id, target_id, channel_id=channel_id)
        return self._descriptions.get(persona, "")

    def _state_key(
        self,
        user_id: int | str,
        target_id: int | str | None,
        channel_id: int | str | None,
    ) -> str:
        return f"{user_id}:{target_id}:{channel_id}"

    def _select_persona(
        self,
        score: float,
        traits: dict[str, float],
        relationship: str | None,
        key: str,
        average_sentiment: float | None,
    ) -> str:
        friendly_score = score + traits.get("friendly", 0)
        playful_score = score + traits.get("playful", 0)
        last_persona = self._persona_state.get(key)
        last_relationship = self._relationship_state.get(key)
        last_sentiment = self._sentiment_state.get(key)
        relationship_persona = {"friend": "friendly", "rival": "snarky"}.get(
            relationship
        )
        apply_hysteresis = True
        if relationship != last_relationship:
            last_persona = relationship_persona
        if (
            apply_hysteresis
            and self._sentiment_hysteresis
            and average_sentiment is not None
            and last_sentiment is not None
            and abs(average_sentiment - last_sentiment) > self._sentiment_hysteresis
        ):
            apply_hysteresis = False
        if apply_hysteresis and last_persona:
            stabilized = self._apply_hysteresis(
                last_persona,
                friendly_score,
                playful_score,
            )
            if stabilized:
                return stabilized
        if friendly_score >= self._friendly:
            return "friendly"
        if playful_score >= self._playful:
            return "playful"
        return "snarky"

    def _apply_hysteresis(
        self,
        last_persona: str,
        friendly_score: float,
        playful_score: float,
    ) -> str | None:
        if last_persona == "friendly":
            if friendly_score >= self._friendly_exit:
                return "friendly"
            return None
        if last_persona == "playful":
            if friendly_score >= self._friendly:
                return "friendly"
            if playful_score >= self._playful_exit:
                return "playful"
            return None
        if last_persona == "snarky":
            if friendly_score < self._friendly and playful_score < self._playful:
                return "snarky"
        return None

    async def _get_sentiment_context(
        self, user_id: int | str, channel_id: int | str | None
    ) -> tuple[float, float | None]:
        if not channel_id or self._sentiment_weight == 0:
            return 0.0, None
        trend = await self._db.get_sentiment_trend(user_id, channel_id)
        if not trend:
            return 0.0, None
        sentiment_sum, message_count = trend
        if not message_count:
            return 0.0, None
        average_sentiment = sentiment_sum / message_count
        return average_sentiment * self._sentiment_weight, average_sentiment
