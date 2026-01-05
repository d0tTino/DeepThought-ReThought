import pytest

sg = pytest.importorskip("examples.social_graph_bot")
if not hasattr(sg, "TrustService"):
    pytest.skip("social_graph_bot optional dependencies not installed", allow_module_level=True)

pytest.importorskip("nats")
from deepthought.services import DBManager, PersonaManager


@pytest.mark.asyncio
async def test_persona_flips_with_channel_sentiment(tmp_path):
    sg.db_manager = DBManager(str(tmp_path / "sg.db"))
    pm = PersonaManager(sg.db_manager, friendly=3, playful=1, sentiment_weight=4)
    user = "u1"
    channel = "c1"

    assert await pm.get_persona(user, channel_id=channel) == "snarky"

    await sg.update_sentiment_trend(user, channel, 0.9)
    assert await pm.get_persona(user, channel_id=channel) == "friendly"

    for _ in range(5):
        await sg.update_sentiment_trend(user, channel, -0.9)

    assert await pm.get_persona(user, channel_id=channel) == "snarky"

    await sg.db_manager.close()
