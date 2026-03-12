import json

import pytest

from deepthought.config import Settings
from deepthought.services.cognitive_core_service import CognitiveCoreService


class DummyNATS:
    is_connected = True


class DummyJS:
    pass


class DummyPublisher:
    def __init__(self):
        self.published = []

    async def publish(self, subject, payload, use_jetstream=True, timeout=10.0):
        self.published.append((subject, payload))


class DummyMemory:
    def __init__(self):
        self.interactions = []

    def store_interaction(self, text):
        self.interactions.append(text)

    def retrieve_context(self, prompt):
        return self.interactions[-3:]


class DummyDB:
    def __init__(self):
        self.memories_by_user = {}

    async def store_memory(self, user_id, memory, topic="", sentiment_score=None):
        self.memories_by_user.setdefault(str(user_id), []).append((topic, memory))

    async def recall_user(self, user_id, limit=None):
        rows = list(reversed(self.memories_by_user.get(str(user_id), [])))
        if limit is not None:
            rows = rows[: int(limit)]
        return rows

    async def close(self):
        return None


class DummyMsg:
    def __init__(self, payload):
        self.data = json.dumps(payload).encode()
        self.acked = False

    async def ack(self):
        self.acked = True


@pytest.mark.asyncio
async def test_memory_lifecycle_consolidation_and_summary_prioritization():
    svc = CognitiveCoreService(
        DummyNATS(), DummyJS(), Settings(), memory=DummyMemory(), db=DummyDB()
    )
    svc._publisher = DummyPublisher()
    svc._consolidation_interval_s = 0.01

    for idx in range(14):
        payload = {
            "user_input": f"small talk {idx}",
            "input_id": f"ep-{idx}",
            "user_id": "u-flow",
        }
        await svc._handle_input(DummyMsg(payload))

    await svc._handle_input(
        DummyMsg(
            {
                "user_input": "My favorite editor is vim and I will follow up later",
                "input_id": "salient-1",
                "user_id": "u-flow",
            }
        )
    )

    assert svc._salience_summaries
    assert svc._tiered_turns["ephemeral"]
    facts = svc._publisher.published[-1][1]["payload"]["retrieved_knowledge"]["facts"]
    assert any("summary:" in fact for fact in facts)
    assert any(
        "tier:long_term" in topic for topic, _ in svc._db.memories_by_user["u-flow"]
    )
