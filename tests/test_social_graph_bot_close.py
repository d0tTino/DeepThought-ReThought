import asyncio
import importlib
import sys
import types

import pytest


class DummyNATS:
    def __init__(self):
        self.closed = False
        self.is_closed = False

    async def close(self):
        self.closed = True
        self.is_closed = True


@pytest.mark.asyncio
async def test_bot_close_cancels_tasks(monkeypatch):
    st = types.ModuleType("sentence_transformers")
    st.SentenceTransformer = lambda *args, **kwargs: None
    st.util = types.SimpleNamespace(cos_sim=lambda a, b: [[0.0]])
    sys.modules.setdefault("sentence_transformers", st)
    sys.modules.setdefault("sentence_transformers.util", st.util)

    sys.modules.pop("examples.social_graph_bot", None)
    sg = importlib.import_module("examples.social_graph_bot")

    closed = False

    class DummyDB:
        async def connect(self):
            pass

        async def init_db(self):
            pass

        async def close(self):
            nonlocal closed
            closed = True

    sg.db_manager = DummyDB()
    monkeypatch.setattr(sg.db_manager, "connect", lambda *a, **k: asyncio.sleep(0))
    monkeypatch.setattr(sg, "init_db", lambda *a, **k: asyncio.sleep(0))

    async def idle_task(*args, **kwargs):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            raise

    monkeypatch.setattr(sg, "monitor_channels", idle_task, raising=False)
    monkeypatch.setattr(sg, "process_deep_reflections", idle_task, raising=False)

    dummy_nats = DummyNATS()
    sg._nats_client = dummy_nats

    bot = sg.SocialGraphBot(monitor_channel_id=1)
    bot._bg_tasks.append(asyncio.create_task(idle_task()))
    bot._bg_tasks.append(asyncio.create_task(idle_task()))
    await asyncio.sleep(0)
    await bot.close()

    assert closed
    assert dummy_nats.closed
