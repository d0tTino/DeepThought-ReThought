"""Finite state machine for quest lifecycle management."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Optional, Set


class QuestState(str, Enum):
    """Enumeration of quest lifecycle states."""

    PROPOSED = "Proposed"
    TRACKED = "Tracked"
    ACTIVE = "Active"
    PAUSED = "Paused"
    BLOCKED = "Blocked"
    SHELVED = "Shelved"
    COMPLETED = "Completed"
    ABANDONED = "Abandoned"


# Mapping of allowed state transitions
ALLOWED_TRANSITIONS: Dict[QuestState, Set[QuestState]] = {
    QuestState.PROPOSED: {QuestState.TRACKED, QuestState.ABANDONED},
    QuestState.TRACKED: {
        QuestState.ACTIVE,
        QuestState.SHELVED,
        QuestState.ABANDONED,
    },
    QuestState.ACTIVE: {
        QuestState.PAUSED,
        QuestState.BLOCKED,
        QuestState.COMPLETED,
        QuestState.ABANDONED,
    },
    QuestState.PAUSED: {QuestState.ACTIVE, QuestState.ABANDONED},
    QuestState.BLOCKED: {QuestState.ACTIVE, QuestState.ABANDONED},
    QuestState.SHELVED: {QuestState.TRACKED, QuestState.ABANDONED},
    QuestState.COMPLETED: set(),  # terminal
    QuestState.ABANDONED: set(),  # terminal
}


@dataclass
class QuestFSM:
    """State machine with TTL-based auto-pruning."""

    state: QuestState = QuestState.PROPOSED
    ttl_seconds: Optional[int] = None
    _last_refresh: datetime = datetime.utcnow()

    def transition(self, new_state: QuestState, *, now: Optional[datetime] = None) -> None:
        """Transition to ``new_state`` if allowed.

        ``now`` may be provided to override the current time for testing.
        """

        self.prune(now=now)
        if self.state == QuestState.ABANDONED:
            raise ValueError("Cannot transition from ABANDONED state")
        allowed = ALLOWED_TRANSITIONS[self.state]
        if new_state not in allowed:
            raise ValueError(f"Invalid transition from {self.state} to {new_state}")
        self.state = new_state
        self.refresh(now=now)

    def refresh(self, *, now: Optional[datetime] = None) -> None:
        """Refresh the TTL timer."""

        self._last_refresh = now or datetime.utcnow()

    def is_expired(self, *, now: Optional[datetime] = None) -> bool:
        """Return ``True`` if the FSM has exceeded its TTL."""

        if self.ttl_seconds is None:
            return False
        now = now or datetime.utcnow()
        return now - self._last_refresh > timedelta(seconds=self.ttl_seconds)

    def prune(self, *, now: Optional[datetime] = None) -> None:
        """Auto-transition to ``ABANDONED`` if the TTL has expired."""

        if self.state in {QuestState.COMPLETED, QuestState.ABANDONED}:
            return
        if self.is_expired(now=now):
            self.state = QuestState.ABANDONED

    # ------------------------------------------------------------------
    def ttl_metadata(self) -> tuple[Optional[int], datetime]:
        """Return TTL seconds and last refresh timestamp.

        These values are useful for external schedulers that need to reason
        about when the FSM should expire without mutating it.
        """

        return self.ttl_seconds, self._last_refresh

    def expires_at(self) -> Optional[datetime]:
        """Return the absolute expiration time if the FSM has a TTL.

        ``None`` is returned when no TTL is configured.
        """

        if self.ttl_seconds is None:
            return None
        return self._last_refresh + timedelta(seconds=self.ttl_seconds)
