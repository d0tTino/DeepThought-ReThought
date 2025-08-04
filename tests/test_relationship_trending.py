import pytest

pytest.importorskip("aiosqlite")
pytest.importorskip("nats")

from deepthought.services import DBManager


@pytest.mark.asyncio
async def test_update_and_get_relationship_trend(tmp_path):
    manager = DBManager(str(tmp_path / "db.sqlite"))
    await manager.init_db()

    await manager.update_relationship_trend("u1", "u2", 0.4)
    await manager.update_relationship_trend("u1", "u2", -0.2)

    avg, count = await manager.get_relationship_trend("u1", "u2")
    assert avg == pytest.approx(0.1)
    assert count == 2

    await manager.close()


@pytest.mark.asyncio
async def test_update_relationship_trend_validation(tmp_path):
    manager = DBManager(str(tmp_path / "db.sqlite"))
    await manager.init_db()

    with pytest.raises(ValueError):
        await manager.update_relationship_trend("u1", "u2", "bad")

    with pytest.raises(ValueError):
        await manager.update_relationship_trend("u1", "u2", 1.5)

    await manager.close()
