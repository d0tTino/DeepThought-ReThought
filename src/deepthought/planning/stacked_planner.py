from __future__ import annotations

"""Stacked planner with basic Belief-Desire-Intention layers."""

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Iterable, List

import requests

logger = logging.getLogger(__name__)


class DiscordThoughtWriter:
    """Send planner summaries to a Discord channel."""

    def __init__(
        self, channel_id: str | None = None, token: str | None = None
    ) -> None:
        self._channel_id = channel_id or os.getenv("THOUGHT_CHANNEL")
        self._token = token or os.getenv("DISCORD_TOKEN")

    def send(self, summary: dict) -> None:
        if not self._channel_id or not self._token:
            return
        url = (
            f"https://discord.com/api/v10/channels/{self._channel_id}/messages"
        )
        headers = {"Authorization": f"Bot {self._token}"}
        payload = {"content": json.dumps(summary)}
        try:  # pragma: no cover - network operations
            requests.post(url, headers=headers, json=payload, timeout=5)
        except Exception:  # pragma: no cover - failure is non-fatal
            logger.warning("Failed to send summary to Thought Server", exc_info=True)


class StackedPlanner:
    """Planner with reactive, investigative and arc layers."""

    def __init__(
        self,
        translator: object,
        planner_fn: Callable[[str, str], List[str]],
        snapshot_dir: str | Path = "planner_snapshots",
        *,
        thought_writer: DiscordThoughtWriter | None = None,
    ) -> None:
        self._translator = translator
        self._planner_fn = planner_fn
        self._snapshot_dir = Path(snapshot_dir)
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._writer = thought_writer or DiscordThoughtWriter()
        self._beliefs: List[str] = []
        self._desires: List[str] = []
        self._intentions: List[str] = []
        self._last_summary = datetime.utcnow()

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
    def utility_score(
        self,
        *,
        info_gain: float = 0.0,
        social_capital: float = 0.0,
        cover_risk: float = 0.0,
        effort: float = 0.0,
    ) -> float:
        return info_gain + social_capital - cover_risk - effort

    def should_act(
        self,
        *,
        info_gain: float = 0.0,
        social_capital: float = 0.0,
        cover_risk: float = 0.0,
        effort: float = 0.0,
    ) -> bool:
        return (
            self.utility_score(
                info_gain=info_gain,
                social_capital=social_capital,
                cover_risk=cover_risk,
                effort=effort,
            )
            >= 0.0
        )

    # ------------------------------------------------------------------
    # Layer processing
    def _reactive_layer(self, actions: List[str]) -> List[str]:
        return actions

    def _investigative_layer(self, actions: List[str]) -> List[str]:
        return actions

    def _arc_layer(self, actions: List[str]) -> List[str]:
        return actions

    # ------------------------------------------------------------------
    def generate_plan(self, goal: str) -> List[str]:
        domain, problem = self._translator.translate(goal)
        actions = self._planner_fn(domain, problem)
        actions = self._reactive_layer(actions)
        actions = self._investigative_layer(actions)
        actions = self._arc_layer(actions)

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
                self._writer.send(summary)
            except Exception:  # pragma: no cover - network errors
                logger.warning("Failed to publish planner summary", exc_info=True)
            self._last_summary = now

