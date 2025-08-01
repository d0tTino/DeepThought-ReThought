import importlib
import os
import sys
import types

import pytest

pytest.importorskip("discord")


def reload_sg(monkeypatch):
    monkeypatch.setitem(os.environ, "ALLOW_DECEPTION", "1")
    import tests.test_deceptive_reply as base

    sys.modules.pop("examples.social_graph_bot", None)
    return importlib.import_module("examples.social_graph_bot"), base.DummyMessage


@pytest.mark.asyncio
async def test_dynamic_deceptive_reply(monkeypatch, tmp_path):
    sg, DummyMessage = reload_sg(monkeypatch)
    sg.db_manager = sg.DBManager(str(tmp_path / "sg.db"))
    await sg.db_manager.connect()
    await sg.db_manager.init_db()

    call_count = {"n": 0}

    def fake_pipeline(task, model=None):
        assert task == "text-generation"

        def gen(prompt, **kwargs):
            call_count["n"] += 1
            return [{"generated_text": f"lie{call_count['n']}"}]

        return gen

    if "transformers" not in sys.modules:
        sys.modules["transformers"] = types.ModuleType("transformers")
    monkeypatch.setattr(
        sys.modules["transformers"], "pipeline", fake_pipeline, raising=False
    )

    r1 = await sg.maybe_deceptive_reply(1, "what are your plans?")
    r2 = await sg.maybe_deceptive_reply(1, "what are your goals?")

    assert r1 != r2
    assert await sg.db_manager.get_last_lie(1, "what are your plans?") == r1
    assert await sg.db_manager.get_last_lie(1, "what are your goals?") == r2

    r1b = await sg.maybe_deceptive_reply(1, "what are your plans?")
    assert r1b == r1
    assert call_count["n"] == 2

    await sg.db_manager.close()
