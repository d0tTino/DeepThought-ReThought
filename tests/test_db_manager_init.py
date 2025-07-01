import aiosqlite
import pytest

from deepthought.services import DBManager

TABLES = [
    "interactions",
    "affinity",
    "memories",
    "theories",
    "queued_tasks",
    "sentiment_trends",
    "themes",
    "user_flags",
    "recent_topics",
]


@pytest.mark.asyncio
async def test_db_manager_init_creates_tables_once(tmp_path):
    db_file = tmp_path / "db.sqlite"
    manager = DBManager(str(db_file))
    await manager.init_db()

    async with aiosqlite.connect(str(db_file)) as db:
        for table in TABLES:
            async with db.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            ) as cur:
                row = await cur.fetchone()
            assert row[0] == 1, f"{table} table should exist exactly once"

    await manager.close()
