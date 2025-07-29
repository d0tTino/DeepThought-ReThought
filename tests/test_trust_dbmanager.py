import pytest

pytest.importorskip("aiosqlite")

from deepthought.services.db_manager import DBManager


@pytest.mark.asyncio
async def test_adjust_get_trust(tmp_path):
    db_file = tmp_path / "db.sqlite"
    manager = DBManager(str(db_file))
    await manager.init_db()

    await manager.adjust_trust("u1", 0.5)
    await manager.adjust_trust("u1", -0.2)

    assert pytest.approx(await manager.get_trust("u1")) == 0.3

    await manager.close()
