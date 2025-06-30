import aiosqlite
import pytest

import examples.social_graph_bot as sg


@pytest.mark.asyncio
async def test_relationship_table_and_updates(tmp_path):
    sg.db_manager = sg.DBManager(str(tmp_path / "sg.db"))
    await sg.db_manager.connect()
    await sg.init_db()

    async with aiosqlite.connect(str(tmp_path / "sg.db")) as db:
        async with db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='relationships'"
        ) as cur:
            row = await cur.fetchone()
    assert row is not None, "relationships table should exist"

    await sg.log_interaction("u1", "u2", sentiment_score=0.3)
    await sg.log_interaction("u1", "u2", sentiment_score=0.2)
    await sg.log_interaction("u1")

    async with aiosqlite.connect(str(tmp_path / "sg.db")) as db:
        async with db.execute(
            "SELECT interaction_count, sentiment_sum FROM relationships WHERE source_id=? AND target_id=?",
            ("u1", "u2"),
        ) as cur:
            row = await cur.fetchone()
    assert row == (2, 0.5)

    friendliness = await sg.get_friendliness("u1", "u2")
    assert pytest.approx(friendliness) == 0.25
    assert await sg.get_hostility("u1", "u2") == 0.0

    await sg.log_interaction("u2", "u1", sentiment_score=-1.0)
    assert await sg.get_friendliness("u2", "u1") == 0.0
    assert await sg.get_hostility("u2", "u1") == -1.0

    await sg.db_manager.close()
