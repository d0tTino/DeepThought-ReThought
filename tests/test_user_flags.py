import aiosqlite
import pytest

import deepthought.social_graph as sg


@pytest.mark.asyncio
async def test_user_flags_table_and_functions(tmp_path):
    sg.db_manager = sg.DBManager(str(tmp_path / "sg.db"))
    await sg.db_manager.connect()
    await sg.init_db()

    async with aiosqlite.connect(str(tmp_path / "sg.db")) as db:
        async with db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='user_flags'"
        ) as cur:
            row = await cur.fetchone()
    assert row is not None, "user_flags table should exist"

    await sg.set_do_not_mock("u1", True)
    assert await sg.is_do_not_mock("u1") is True

    await sg.set_do_not_mock("u1", False)
    assert await sg.is_do_not_mock("u1") is False
    assert await sg.is_do_not_mock("u2") is False
    await sg.db_manager.close()
