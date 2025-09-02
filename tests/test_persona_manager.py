import pytest

sg = pytest.importorskip("examples.social_graph_bot")
if not hasattr(sg, "TrustService"):
    pytest.skip("social_graph_bot optional dependencies not installed", allow_module_level=True)

pytest.importorskip("nats")
from deepthought.services import DBManager, PersonaManager


@pytest.mark.asyncio
async def test_persona_changes_with_affinity(tmp_path):
    sg.db_manager = DBManager(str(tmp_path / "sg.db"))

    pm = PersonaManager(sg.db_manager, friendly=5, playful=2)
    user = "u1"

    assert await pm.get_persona(user) == "snarky"

    await sg.adjust_affinity(user, 2)
    assert await pm.get_persona(user) == "playful"

    await sg.db_manager.adjust_trust(user, 3)
    assert await pm.get_persona(user) == "friendly"

    await sg.db_manager.close()


@pytest.mark.asyncio
async def test_choose_prompt_uses_persona(tmp_path, monkeypatch):
    sg.db_manager = DBManager(str(tmp_path / "sg.db"))

    pm = PersonaManager(sg.db_manager, friendly=2, playful=1)
    user = "u1"
    prompts = {"snarky": ["s"], "playful": ["p"], "friendly": ["f"], "default": ["d"]}

    monkeypatch.setattr("random.choice", lambda opts: opts[0])

    # default persona
    assert await pm.choose_prompt(user, prompts) == "s"

    await sg.adjust_affinity(user, 1)
    assert await pm.choose_prompt(user, prompts) == "p"

    await sg.db_manager.adjust_trust(user, 1)
    assert await pm.choose_prompt(user, prompts) == "f"

    # fallback to default when persona key missing
    assert await pm.choose_prompt(user, {"default": ["x"]}) == "x"

    await sg.db_manager.close()


@pytest.mark.asyncio
async def test_persona_uses_mutual_affinity(tmp_path, monkeypatch):
    sg.db_manager = DBManager(str(tmp_path / "sg.db"))

    pm = PersonaManager(sg.db_manager, friendly=2, playful=1)
    called = False

    async def fake_get_mutual_affinity(_user_id):
        nonlocal called
        called = True
        return 3

    monkeypatch.setattr(sg.db_manager, "get_mutual_affinity", fake_get_mutual_affinity)

    assert await pm.get_persona("u1") == "friendly"
    assert called

    await sg.db_manager.close()
