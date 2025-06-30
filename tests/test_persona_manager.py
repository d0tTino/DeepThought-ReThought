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
async def test_choose_prompt_uses_persona(tmp_path, monkeypatch):
    sg.db_manager = sg.DBManager(str(tmp_path / "sg.db"))
    await sg.db_manager.connect()
    await sg.init_db()

    pm = PersonaManager(sg.db_manager, friendly=2, playful=1)
    user = "u1"
    prompts = {"snarky": ["s"], "playful": ["p"], "friendly": ["f"], "default": ["d"]}

    monkeypatch.setattr("random.choice", lambda opts: opts[0])

    # default persona
    assert await pm.choose_prompt(user, prompts) == "s"

    await sg.adjust_affinity(user, 1)
    assert await pm.choose_prompt(user, prompts) == "p"

    await sg.adjust_affinity(user, 1)
    assert await pm.choose_prompt(user, prompts) == "f"

    # fallback to default when persona key missing
    assert await pm.choose_prompt(user, {"default": ["x"]}) == "x"

    await sg.db_manager.close()
