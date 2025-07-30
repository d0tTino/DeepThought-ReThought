import sys
import types
import pytest

pytest.importorskip("aiosqlite")

sys.modules.setdefault("pyperplan", types.ModuleType("pyperplan"))

from deepthought.services.db_manager import DBManager


@pytest.mark.asyncio
async def test_store_and_get_last_lie(tmp_path):
    db_file = tmp_path / "db.sqlite"
    manager = DBManager(str(db_file))
    await manager.init_db()

    assert await manager.get_last_lie("u1", "q1") is None
    await manager.store_lie("u1", "q1", "r1")
    assert await manager.get_last_lie("u1", "q1") == "r1"

    await manager.store_lie("u1", "q1", "r2")
    assert await manager.get_last_lie("u1", "q1") == "r2"

    await manager.close()

    await manager.close()
