"""Utilities for routing quest updates to communication channels."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)


class QuestWriter:
    """Send quest-related updates to Discord channels.

    Parameters can be provided directly or via environment variables:

    - ``QUEST_BOARD_CHANNEL``: channel for general quest updates.
    - ``QUEST_JOURNAL_TEMPLATE``: template for per-quest journals, e.g. ``"quest-{id}"``.
    - ``OPSSEC_LIES_CHANNEL``: channel for lie ledger updates.
    - ``DAILY_SUMMARY_CHANNEL``: channel for daily planner summaries.
    - ``WHISPERS_DAILY_CHANNEL``: channel for daily whisper summaries.
    - ``ALERTS_CHANNEL``: channel for runtime alerts.
    - ``AUTOPSY_CHANNEL``: channel for quest autopsies.
    - ``PARAMETERS_CHANNEL``: channel for parameter dumps.
    - ``DISCORD_TOKEN``: bot token for authentication.
    """

    def __init__(
        self,
        *,
        board_channel: str | None = None,
        journal_template: str | None = None,
        opssec_channel: str | None = None,
        daily_channel: str | None = None,
        whispers_daily_channel: str | None = None,
        alerts_channel: str | None = None,
        autopsy_channel: str | None = None,
        parameters_channel: str | None = None,
        token: str | None = None,
    ) -> None:
        self._board_channel = board_channel or os.getenv("QUEST_BOARD_CHANNEL")
        self._journal_template = journal_template or os.getenv("QUEST_JOURNAL_TEMPLATE")
        self._opssec_channel = opssec_channel or os.getenv("OPSSEC_LIES_CHANNEL")
        self._daily_channel = daily_channel or os.getenv("DAILY_SUMMARY_CHANNEL")
        self._whispers_daily_channel = whispers_daily_channel or os.getenv("WHISPERS_DAILY_CHANNEL")
        self._alerts_channel = alerts_channel or os.getenv("ALERTS_CHANNEL")
        self._autopsy_channel = autopsy_channel or os.getenv("AUTOPSY_CHANNEL")
        self._parameters_channel = parameters_channel or os.getenv("PARAMETERS_CHANNEL")
        self._token = token or os.getenv("DISCORD_TOKEN")

    # ------------------------------------------------------------------
    def _post(self, channel_id: Optional[str], content: str) -> None:
        if not channel_id or not self._token:
            return
        url = f"https://discord.com/api/v10/channels/{channel_id}/messages"
        headers = {"Authorization": f"Bot {self._token}"}
        payload = {"content": content}
        try:  # pragma: no cover - network operations
            requests.post(url, headers=headers, json=payload, timeout=5)
        except Exception:  # pragma: no cover - failure is non-fatal
            logger.warning("Failed to send message to channel %s", channel_id, exc_info=True)

    # ------------------------------------------------------------------
    def send_board_update(self, quest: Any, event: str = "updated") -> None:
        """Post quest updates to the quest board and journal."""

        message = f"[{event}] {quest.name}: {quest.description}"
        self._post(self._board_channel, message)
        if quest.id is not None:
            self.send_journal_entry(quest.id, message)

    def send_journal_entry(self, quest_id: int, message: str) -> None:
        """Send a note to the per-quest journal."""

        if not self._journal_template:
            return
        channel_id = self._journal_template.format(id=quest_id)
        self._post(channel_id, message)

    def send_lie(self, quest_id: int, lie: str) -> None:
        """Record a lie in the opssec ledger and journal."""

        message = f"Lie recorded for quest {quest_id}: {lie}"
        self._post(self._opssec_channel, message)
        self.send_journal_entry(quest_id, message)

    def send_daily_summary(self, summary: dict) -> None:
        """Publish a daily planner summary."""

        self._post(self._daily_channel, json.dumps(summary))

    def send_whispers_daily(self, message: str) -> None:
        """Publish the daily whispers summary."""

        self._post(self._whispers_daily_channel, message)

    def send_alert(self, message: str) -> None:
        """Send an alert message."""

        self._post(self._alerts_channel, message)

    def send_parameters(self, params: dict) -> None:
        """Dump parameter information."""

        self._post(self._parameters_channel, json.dumps(params))

    def send_quest_story(self, quest: Any, narrative: str) -> None:
        """Publish a quest narrative to the autopsy channel when completed."""

        if getattr(quest, "status", "") != "completed":
            return
        self._post(self._autopsy_channel, narrative)
