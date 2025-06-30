import asyncio
import logging

import aiohttp
import pytest

import examples.social_graph_bot as bot_mod


class DummySession:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, *args, **kwargs):
        raise aiohttp.ClientError("boom")


class TimeoutSession:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, *args, **kwargs):
        raise asyncio.TimeoutError("took too long")


@pytest.mark.asyncio
async def test_send_to_prism_client_error(monkeypatch, caplog):
    monkeypatch.setattr(bot_mod.aiohttp, "ClientSession", lambda: DummySession())
    with caplog.at_level(logging.WARNING):
        await bot_mod.send_to_prism({"x": 1})
    assert any("ClientError" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_send_to_prism_timeout(monkeypatch, caplog):
    monkeypatch.setattr(bot_mod.aiohttp, "ClientSession", lambda: TimeoutSession())
    with caplog.at_level(logging.WARNING):
        await bot_mod.send_to_prism({"x": 1})
    assert any("TimeoutError" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_send_to_prism_timeout_message(monkeypatch, caplog):
    """Ensure the timeout warning includes the expected text."""
    monkeypatch.setattr(bot_mod.aiohttp, "ClientSession", lambda: TimeoutSession())
    with caplog.at_level(logging.WARNING):
        await bot_mod.send_to_prism({"x": 1})
    messages = [rec.getMessage() for rec in caplog.records]
    assert any("TimeoutError sending data to Prism" in msg for msg in messages)
