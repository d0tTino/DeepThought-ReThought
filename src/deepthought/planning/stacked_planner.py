from __future__ import annotations

"""Stacked planner with basic Belief-Desire-Intention layers."""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Iterable, List

from ..bot import interaction as bot_interaction
from ..quest.writer import QuestWriter

logger = logging.getLogger(__name__)


class StackedPlanner:
    """Planner with reactive, investigative and arc layers."""

    UTILITY_WEIGHTS = {
        "info_gain": 1.0,
        "social_capital": 1.0,
        "cover_risk": 1.0,
        "effort": 1.0,
        "vibes_fit": 1.0,
    }
    SILENCE_THRESHOLD = 0.0
    SILENCE_RATE = 5

    def __init__(
        self,
        translator: object,
        planner_fn: Callable[[str, str], List[str]],
        snapshot_dir: str | Path = "planner_snapshots",
        *,
        writer: QuestWriter | None = None,
    ) -> None:
        self._translator = translator
        self._planner_fn = planner_fn
        self._snapshot_dir = Path(snapshot_dir)
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._writer = writer or QuestWriter()
        self._beliefs: List[str] = []
        self._desires: List[str] = []
        self._intentions: List[str] = []
        self._last_summary = datetime.utcnow()
        self._weights = self.UTILITY_WEIGHTS.copy()
        self.silence_threshold = self.SILENCE_THRESHOLD
        self.silence_rate = self.SILENCE_RATE
        self._action_history: List[datetime] = []
        self._next_action_time = datetime.min

    # ------------------------------------------------------------------
    # Beliefs, Desires, Intentions state
    def set_beliefs(self, beliefs: Iterable[str]) -> None:
        self._beliefs = list(beliefs)

    def set_desires(self, desires: Iterable[str]) -> None:
        self._desires = list(desires)

    @property
    def intentions(self) -> List[str]:
        return list(self._intentions)

    # ------------------------------------------------------------------
    # Utility scoring and policy
    def _compute_context_metrics(self, conversation: Iterable[str]) -> dict[str, float]:
        text = " ".join(conversation).lower()
        words = set(text.split())
        info_gain = len(words) / 10.0
        social_capital = 1.0 if any(w in text for w in ["please", "thanks", "thank you"]) else 0.0
        cover_risk = 1.0 if any(w in text for w in ["risky", "danger"]) else 0.0
        effort = len(list(conversation)) / 10.0
        vibes_fit = 1.0 if any(w in text for w in ["vibe", "cool", "awesome"]) else 0.0
        return {
            "info_gain": info_gain,
            "social_capital": social_capital,
            "cover_risk": cover_risk,
            "effort": effort,
            "vibes_fit": vibes_fit,
        }

    def utility_score(self, conversation: Iterable[str] | None = None) -> float:
        metrics = self._compute_context_metrics(conversation or [])
        w = self._weights
        return (
            metrics["info_gain"] * w["info_gain"]
            + metrics["social_capital"] * w["social_capital"]
            + metrics["vibes_fit"] * w["vibes_fit"]
            - metrics["cover_risk"] * w["cover_risk"]
            - metrics["effort"] * w["effort"]
        )

    def should_act(
        self,
        conversation: Iterable[str] | None = None,
        *,
        cooldown: float = 0.0,
        participants: Iterable[object] | None = None,
        bot_threshold: int = 2,
    ) -> bool:
        now = datetime.utcnow()
        if now < self._next_action_time:
            return False
        if participants and bot_interaction.is_crowded(participants, bot_threshold):
            self._next_action_time = now + timedelta(seconds=cooldown)
            return False
        score = self.utility_score(conversation or [])
        threshold = self.silence_threshold
        self._action_history = [t for t in self._action_history if now - t < timedelta(minutes=1)]
        if len(self._action_history) >= self.silence_rate:
            threshold += 1.0
        self._next_action_time = now + timedelta(seconds=cooldown)
        if score >= threshold:
            self._action_history.append(now)
            return True
        return False

    # ------------------------------------------------------------------
    # Layer processing
    def _reactive_layer(self, actions: List[str]) -> List[str]:
        """First pass to handle immediate reactions.

        Each action is prefixed with ``"react:"`` so later stages can
        differentiate work performed at this layer.
        """

        return [f"react:{act}" for act in actions]

    def _investigative_layer(self, actions: List[str]) -> List[str]:
        """Second pass to gather more information.

        Actions are turned into questions by appending ``"?"``.
        """

        return [f"{act}?" for act in actions]

    def _arc_layer(self, actions: List[str]) -> List[str]:
        """Final pass to craft a longer term narrative arc.

        Actions are capitalised to represent committed intentions.
        """

        return [act.upper() for act in actions]

    # ------------------------------------------------------------------
    def generate_plan(self, goal: str) -> List[str]:
        domain, problem = self._translator.translate(goal)
        actions = self._planner_fn(domain, problem)
        actions = self._reactive_layer(actions)
        self._persist_layer_snapshot(goal, actions, "reactive")
        actions = self._investigative_layer(actions)
        self._persist_layer_snapshot(goal, actions, "investigative")
        actions = self._arc_layer(actions)
        self._persist_layer_snapshot(goal, actions, "arc")

        filtered: List[str] = []
        for act in actions:
            if self.should_act():
                filtered.append(act)

        self._intentions.extend(filtered)
        self._persist_snapshot(goal, filtered)
        self._maybe_send_summary()
        return filtered

    # ------------------------------------------------------------------
    def _persist_snapshot(self, goal: str, actions: List[str]) -> None:
        snapshot = {
            "timestamp": datetime.utcnow().isoformat(),
            "goal": goal,
            "beliefs": self._beliefs,
            "desires": self._desires,
            "intentions": actions,
        }
        fname = datetime.utcnow().strftime("%Y%m%d%H%M%S.json")
        path = self._snapshot_dir / fname
        try:
            path.write_text(json.dumps(snapshot), encoding="utf-8")
        except Exception:  # pragma: no cover - disk errors
            logger.warning("Failed to write planner snapshot", exc_info=True)

    def _persist_layer_snapshot(self, goal: str, actions: List[str], layer: str) -> None:
        """Persist the state of a single planning layer.

        Parameters
        ----------
        goal:
            The goal being pursued.
        actions:
            The list of actions after the layer transformation.
        layer:
            Name of the layer (``reactive``, ``investigative`` or ``arc``).
        """

        snapshot = {
            "timestamp": datetime.utcnow().isoformat(),
            "goal": goal,
            "layer": layer,
            "actions": actions,
        }
        fname = datetime.utcnow().strftime("%Y%m%d%H%M%S%f") + f"_{layer}.json"
        path = self._snapshot_dir / fname
        try:
            path.write_text(json.dumps(snapshot), encoding="utf-8")
        except Exception:  # pragma: no cover - disk errors
            logger.warning("Failed to write %s layer snapshot", layer, exc_info=True)

    def _maybe_send_summary(self) -> None:
        now = datetime.utcnow()
        if now - self._last_summary >= timedelta(hours=1):
            summary = {
                "timestamp": now.isoformat(),
                "beliefs": self._beliefs,
                "desires": self._desires,
                "intentions": self._intentions,
            }
            try:
                self._writer.send_daily_summary(summary)
            except Exception:  # pragma: no cover - network errors
                logger.warning("Failed to publish planner summary", exc_info=True)
            self._last_summary = now
