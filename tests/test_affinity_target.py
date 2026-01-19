import pytest

pytest.importorskip("aiosqlite")
import aiosqlite

from deepthought.services import DBManager


@pytest.mark.asyncio
async def test_adjust_affinity_with_target_creates_relationship(tmp_path):
    db_path = tmp_path / "sg.db"
    db = DBManager(str(db_path))
    await db.connect()
    await db.init_db()

    user_a = 101
    user_b = 202
    assert await db.get_pair_mutual_affinity(user_a, user_b) == 0.0

    await db.adjust_affinity(user_a, 1.0, target_id=user_b)

    assert await db.get_pair_mutual_affinity(user_a, user_b) != 0.0

    async with aiosqlite.connect(str(db_path)) as conn:
        async with conn.execute(
            "SELECT source_id, target_id FROM relationships WHERE source_id=? AND target_id=?",
            (str(user_a), str(user_b)),
        ) as cur:
            row = await cur.fetchone()
        async with conn.execute(
            "SELECT source_id FROM relationships WHERE source_id=? AND target_id=?",
            (str(user_b), str(user_a)),
        ) as cur:
            reverse_row = await cur.fetchone()

    assert row == (str(user_a), str(user_b))
    assert reverse_row is None
