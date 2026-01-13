import pytest

sg = pytest.importorskip("examples.social_graph_bot")
if not hasattr(sg, "TrustService"):
    pytest.skip("social_graph_bot optional dependencies not installed", allow_module_level=True)
from deepthought.services import DBManager, PersonaManager

pytest.importorskip("nats")


@pytest.mark.asyncio
async def test_personality_traits_override_affinity(tmp_path):
    sg.db_manager = DBManager(str(tmp_path / "sg.db"))
    pm = PersonaManager(sg.db_manager, friendly=5, playful=2)
    user = "u1"

    # Default persona is snarky when score is low
    assert await pm.get_persona(user) == "snarky"

    # Personality can push persona to friendly
    await pm.update_personality(user, {"friendly": 5})
    assert await pm.get_persona(user) == "friendly"

    # Updating traits can shift persona again
    await pm.update_personality(user, {"friendly": 0, "playful": 2})
    assert await pm.get_persona(user) == "playful"

    await sg.db_manager.close()


@pytest.mark.asyncio
async def test_personality_combines_with_affinity(tmp_path):
    sg.db_manager = DBManager(str(tmp_path / "sg.db"))
    pm = PersonaManager(sg.db_manager, friendly=5, playful=2)
    user = "u2"

    await sg.adjust_affinity(user, 1)
    assert await pm.get_persona(user) == "snarky"

    await pm.update_personality(user, {"playful": 1})
    assert await pm.get_persona(user) == "playful"

    await sg.db_manager.close()
