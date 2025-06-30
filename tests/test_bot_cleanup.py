import asyncio

import discord
import pytest

import deepthought.social_graph as sg
import examples.social_graph_bot as bot_mod


class DummyNATS:
    def __init__(self):
        self.closed = False
        self.is_closed = False

    async def close(self):
        self.closed = True
        self.is_closed = True


@pytest.mark.asyncio
async def test_bot_cleanup_on_cancel(tmp_path, monkeypatch):
    sg.db_manager = sg.DBManager(str(tmp_path / "sg.db"))
    await sg.db_manager.connect()
    await sg.init_db()

    dummy_nats = DummyNATS()
    sg._nats_client = dummy_nats
    sg._js_context = object()
    sg._input_publisher = object()

    async def dummy_close(self):
        pass

    async def dummy_start(self, *args, **kwargs):
        await asyncio.Future()

    monkeypatch.setattr(discord.Client, "close", dummy_close, raising=False)
    monkeypatch.setattr(discord.Client, "start", dummy_start, raising=False)

    bot = bot_mod.SocialGraphBot(monitor_channel_id=1)
    task = asyncio.create_task(bot.start("token"))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    await bot.close()

    assert sg.db_manager._db is None
    assert dummy_nats.closed
    assert sg._nats_client is None
    assert sg._js_context is None
    assert sg._input_publisher is None
