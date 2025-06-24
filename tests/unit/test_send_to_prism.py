import logging

import aiohttp
import asyncio
import pytest

import examples.social_graph_bot as sg


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
    monkeypatch.setattr(sg.aiohttp, "ClientSession", lambda: DummySession())
    with caplog.at_level(logging.WARNING):
        await sg.send_to_prism({"x": 1})
    assert any("ClientError" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_send_to_prism_timeout(monkeypatch, caplog):
    monkeypatch.setattr(sg.aiohttp, "ClientSession", lambda: TimeoutSession())
    with caplog.at_level(logging.WARNING):
        await sg.send_to_prism({"x": 1})
    assert any("TimeoutError" in r.getMessage() for r in caplog.records)
