from __future__ import annotations

import asyncio
import random
from datetime import datetime, timezone
from typing import Any

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
        self._social_persona_state: dict[str, dict[str, Any]] = {}

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

    async def update_personality(self, user_id: int, traits: dict[str, float]) -> None:
        """Update stored personality ``traits`` for ``user_id``.

        The provided ``traits`` mapping is merged with any existing traits for
        the user, overwriting values for matching keys. Trait maps are stored
        as JSON dictionaries in the ``user_profiles`` table.
        """

        await self._ensure_initialized()
        uid = str(user_id)
        current = self._personality.get(uid)
        if current is None:
            current = await self._load_user_traits(user_id)
        current.update(self._normalize_traits(traits))
        self._personality[uid] = current
        await self._store_user_traits(user_id, current)

    async def get_persona(
        self,
        user_id: int | str,
        target_id: int | str | None = None,
        *,
        channel_id: int | str | None = None,
    ) -> str:
        await self._ensure_initialized()
        uid = str(user_id)
        traits = self._personality.get(uid)
        if traits is None:
            traits = await self._load_user_traits(user_id)
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

    async def transition_persona_state(
        self,
        user_id: int | str,
        *,
        signals: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Transition durable persona-state using social and feedback signals."""
        await self._ensure_initialized()
        uid = str(user_id)
        profile = await self._db.get_user_profile(uid)
        model = dict(profile) if isinstance(profile, dict) else {}

        state_block = model.get("persona_state")
        state_data = dict(state_block) if isinstance(state_block, dict) else {}
        current_state = str(state_data.get("current") or "new_acquaintance")
        current_state = self._normalize_social_state(current_state)
        evidence_log = state_data.get("evidence")
        evidence = list(evidence_log) if isinstance(evidence_log, list) else []

        normalized = dict(signals or {})
        next_state, reason, policy_hints = self._derive_social_state(
            current_state,
            normalized,
        )
        timestamp = datetime.now(timezone.utc).isoformat()
        evidence_entry = {
            "at": timestamp,
            "state": next_state,
            "reason": reason,
            "signals": {
                "delta": normalized.get("delta"),
                "affinity": normalized.get("affinity"),
                "trust": normalized.get("trust"),
                "feedback": normalized.get("feedback"),
                "perception": normalized.get("perception"),
                "familiarity_tier": normalized.get("familiarity_tier"),
                "relationship_status": normalized.get("relationship_status"),
            },
        }
        evidence.append(evidence_entry)
        evidence = evidence[-20:]

        persona_state = {
            "current": next_state,
            "updated_at": timestamp,
            "reason": reason,
            "policy_hints": policy_hints,
            "evidence": evidence,
        }
        model["persona_state"] = persona_state
        await self._db.set_user_profile(uid, model)
        self._social_persona_state[uid] = persona_state
        return persona_state

    @staticmethod
    def _normalize_social_state(state: str) -> str:
        allowed = {
            "new_acquaintance",
            "familiar",
            "trusted",
            "repair_mode",
            "uncertain_mode",
        }
        return state if state in allowed else "new_acquaintance"

    def _derive_social_state(
        self,
        current_state: str,
        signals: dict[str, Any],
    ) -> tuple[str, str, dict[str, Any]]:
        perception = signals.get("perception")
        perception_map = perception if isinstance(perception, dict) else {}
        manipulation = float(perception_map.get("manipulation", 0.0) or 0.0)
        avoidance = float(perception_map.get("avoidance", 0.0) or 0.0)
        flirtation = float(perception_map.get("flirtation", 0.0) or 0.0)
        delta = float(signals.get("delta", 0.0) or 0.0)
        trust = float(signals.get("trust", 0.0) or 0.0)
        affinity = float(signals.get("affinity", 0.0) or 0.0)
        familiarity_tier = str(signals.get("familiarity_tier") or "low")
        relationship_status = str(signals.get("relationship_status") or "neutral")
        feedback = signals.get("feedback")
        feedback_map = feedback if isinstance(feedback, dict) else {}
        negative_feedback = bool(feedback_map.get("negative"))
        positive_feedback = bool(feedback_map.get("positive"))
        uncertain = bool(feedback_map.get("uncertain")) or bool(signals.get("low_confidence"))

        if manipulation >= 0.45 or avoidance >= 0.65 or negative_feedback:
            next_state = "repair_mode"
            reason = "social_or_feedback_repair"
        elif uncertain:
            next_state = "uncertain_mode"
            reason = "uncertainty_signal"
        elif trust >= 6.0 or affinity >= 6.0 or relationship_status == "friend":
            next_state = "trusted"
            reason = "high_trust_or_affinity"
        elif familiarity_tier in {"medium", "high"} or affinity >= 2.0:
            next_state = "familiar"
            reason = "growing_familiarity"
        else:
            next_state = "new_acquaintance"
            reason = "limited_history"

        if current_state == "repair_mode" and next_state in {"familiar", "trusted"} and not positive_feedback and delta <= 0:
            next_state = "repair_mode"
            reason = "repair_hold"

        policy_hints = {
            "tone": {
                "new_acquaintance": "polite_clear",
                "familiar": "warm_consistent",
                "trusted": "direct_collaborative",
                "repair_mode": "careful_empathic",
                "uncertain_mode": "clarifying",
            }[next_state],
            "allow_playfulness": next_state in {"familiar", "trusted"} and flirtation > 0.2,
            "prioritize_clarification": next_state in {"repair_mode", "uncertain_mode"},
            "avoid_assumptions": next_state == "uncertain_mode",
            "repair_needed": next_state == "repair_mode",
        }
        return next_state, reason, policy_hints

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

    async def _ensure_initialized(self) -> None:
        if self._init_task is not None:
            await self._init_task
            self._init_task = None

    async def _load_user_traits(self, user_id: int | str) -> dict[str, float]:
        profile = await self._db.get_user_profile(user_id)
        traits = self._normalize_traits(profile)
        self._personality[str(user_id)] = traits
        return traits

    async def _store_user_traits(self, user_id: int | str, traits: dict[str, float]) -> None:
        await self._db.set_user_profile(user_id, traits)

    def _normalize_traits(self, traits: object | None) -> dict[str, float]:
        if not isinstance(traits, dict):
            return {}
        if "traits" in traits and isinstance(traits["traits"], dict):
            traits = traits["traits"]
        normalized: dict[str, float] = {}
        for key, value in traits.items():
            try:
                normalized[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        return normalized
