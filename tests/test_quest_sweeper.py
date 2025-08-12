from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import pytest

from deepthought.quest.fsm import QuestFSM, QuestState
from deepthought.quest.sweeper import sweep_expired_quests


@dataclass
class Quest:
    id: int
    name: str
    description: str
    status: str


class DummyStorage:
    def __init__(self, quests):
        self._quests = quests
        self.updated = []

    async def list_quests(self):
        return list(self._quests)

    async def update_quest(self, quest):
        self.updated.append(quest)


class DummyWriter:
    def __init__(self):
        self.messages = []

    def send_board_update(self, payload, event="updated"):
        self.messages.append((payload, event))


@pytest.mark.asyncio
async def test_sweep_expired_moves_and_summarizes():
    base = datetime(2024, 1, 1)
    expired_fsm = QuestFSM(ttl_seconds=10, _last_refresh=base)
    active_fsm = QuestFSM(ttl_seconds=100, _last_refresh=base)

    expired = Quest(1, "old", "", QuestState.TRACKED.value)
    active = Quest(2, "new", "", QuestState.TRACKED.value)

    storage = DummyStorage([(expired, expired_fsm), (active, active_fsm)])
    writer = DummyWriter()

    moved = await sweep_expired_quests(storage, writer=writer, now=base + timedelta(seconds=20))

    assert moved == [expired]
    assert expired.status == QuestState.ABANDONED.value
    assert writer.messages[0][0] == {"abandoned": [1]}
    # ensure TTL hook works
    assert expired_fsm.expires_at() == base + timedelta(seconds=10)
