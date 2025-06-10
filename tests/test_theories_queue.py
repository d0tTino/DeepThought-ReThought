
import aiosqlite
import pytest

import examples.social_graph_bot as sg


@pytest.mark.asyncio
async def test_store_theory(tmp_path):
    db_file = tmp_path / "db.sqlite"
    sg.DB_PATH = str(db_file)
    await sg.init_db()
    await sg.store_theory("u1", "insomniac", 0.5)
    async with aiosqlite.connect(sg.DB_PATH) as db:
        async with db.execute("SELECT theory FROM theories WHERE subject_id=?", ("u1",)) as cur:
            row = await cur.fetchone()
    assert row[0] == "insomniac"


@pytest.mark.asyncio
async def test_store_theory_update(tmp_path):
    db_file = tmp_path / "db.sqlite"
    sg.DB_PATH = str(db_file)
    await sg.init_db()
    await sg.store_theory("u1", "insomniac", 0.5)
    await sg.store_theory("u1", "insomniac", 0.8)
    async with aiosqlite.connect(sg.DB_PATH) as db:
        async with db.execute(
            "SELECT confidence FROM theories WHERE subject_id=? AND theory=?",
            ("u1", "insomniac"),
        ) as cur:
            row = await cur.fetchone()
    assert row[0] == 0.8


@pytest.mark.asyncio
async def test_queue_deep_reflection(tmp_path):
    db_file = tmp_path / "db.sqlite"
    sg.DB_PATH = str(db_file)
    await sg.init_db()
    task_id = await sg.queue_deep_reflection("u2", {"channel_id": 1}, "hello")
    async with aiosqlite.connect(sg.DB_PATH) as db:
        async with db.execute("SELECT status FROM queued_tasks WHERE task_id=?", (task_id,)) as cur:
            row = await cur.fetchone()
    assert row[0] == "pending"


def test_evaluate_triggers():
    class Dummy:
        def __init__(self, content, hour):
            from datetime import datetime, timezone

            self.content = content
            self.created_at = datetime(2025, 1, 1, hour, tzinfo=timezone.utc)

    m1 = Dummy("I agree with you", 1)
    assert ("social chameleon", 0.6) in sg.evaluate_triggers(m1)
    m2 = Dummy("hello", 2)
    assert ("insomniac", 0.7) in sg.evaluate_triggers(m2)

