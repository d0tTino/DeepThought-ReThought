import pytest

pytest.importorskip("aiosqlite")

from deepthought.eda.events import EventSubjects
from deepthought.services.scheduler import SchedulerService


class DummyPublisher:
    def __init__(self):
        self.published = []

    async def publish(self, subject, payload, use_jetstream=True, timeout=10.0):
        self.published.append((subject, payload))


class DummyMemoryDAL:
    def get_recent_facts(self, count=3):
        return []


class DummyGraphDAL:
    def add_entity(self, label, props):
        return None


class DummySummaryDB:
    def __init__(self):
        self.upserts = []

    async def list_users_with_long_history(self, min_memories):
        assert min_memories == 2
        return [("42", 3)]

    async def recall_user(self, user_id, limit=None):
        return [("topic", "User likes chess"), ("topic", "User prefers concise answers")]

    async def upsert_user_summary(self, user_id, summary, source_count):
        self.upserts.append((user_id, summary, source_count))


@pytest.mark.asyncio
async def test_generate_user_history_summaries_publishes_refresh_event():
    service = SchedulerService(
        DummyPublisher(),
        DummyMemoryDAL(),
        DummyGraphDAL(),
        user_summary_interval=100.0,
        min_user_history_for_summary=2,
        summary_db=DummySummaryDB(),
    )

    await service._generate_user_history_summaries()

    assert service._summary_db.upserts
    assert any(subject == EventSubjects.USER_SUMMARY_REFRESH for subject, _payload in service._publisher.published)
