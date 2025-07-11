import sys
import types

import pytest

pytest.importorskip("aiosqlite")
pytest.importorskip("nats")
import aiosqlite

sys.modules.setdefault("faiss", types.ModuleType("faiss"))

from deepthought.services.db_manager import DBManager

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


def test_create_table_statements_returns_class_queries(tmp_path):
    manager = DBManager(str(tmp_path / "db.sqlite"))
    assert manager._create_table_statements() is DBManager.CREATE_TABLE_QUERIES
