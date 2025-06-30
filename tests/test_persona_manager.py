import pytest

import examples.social_graph_bot as sg
from deepthought.services import PersonaManager


@pytest.mark.asyncio
async def test_persona_changes_with_affinity(tmp_path):
    sg.db_manager = sg.DBManager(str(tmp_path / "sg.db"))
    await sg.db_manager.connect()
    await sg.init_db()

    pm = PersonaManager(sg.db_manager, friendly=5, playful=2)
    user = "u1"

    assert await pm.get_persona(user) == "snarky"

    await sg.adjust_affinity(user, 2)
    assert await pm.get_persona(user) == "playful"

    await sg.adjust_affinity(user, 3)
    assert await pm.get_persona(user) == "friendly"

    await sg.db_manager.close()


@pytest.mark.asyncio
async def test_affinity_adjustment_new_db(tmp_path):
    sg.db_manager = sg.DBManager(str(tmp_path / "sg_new.db"))
    await sg.db_manager.connect()
    await sg.init_db()

    user = "user1"
    await sg.adjust_affinity(user, 1)
    score = await sg.get_affinity(user)
    assert score == 1

    await sg.db_manager.close()
