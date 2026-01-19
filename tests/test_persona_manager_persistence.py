import pytest

from deepthought.services import DBManager, PersonaManager


@pytest.mark.asyncio
async def test_persona_traits_persist_across_manager_instances(tmp_path):
    db_path = tmp_path / "sg.db"
    db_manager = DBManager(str(db_path))
    pm = PersonaManager(db_manager, friendly=5, playful=2)
    user = "u1"

    await pm.update_personality(user, {"friendly": 5})
    assert await pm.get_persona(user) == "friendly"
    await db_manager.close()

    fresh_manager = DBManager(str(db_path))
    fresh_pm = PersonaManager(fresh_manager, friendly=5, playful=2)

    assert await fresh_pm.get_persona(user) == "friendly"
    await fresh_manager.close()


@pytest.mark.asyncio
async def test_persona_flips_with_traits_and_affinity(tmp_path):
    db_manager = DBManager(str(tmp_path / "sg.db"))
    pm = PersonaManager(db_manager, friendly=5, playful=2)
    user = "u2"

    assert await pm.get_persona(user) == "snarky"

    await pm.update_personality(user, {"friendly": 4})
    assert await pm.get_persona(user) == "snarky"

    await db_manager.adjust_affinity(user, 1)
    assert await pm.get_persona(user) == "friendly"

    await db_manager.close()
