import logging

import aiohttp
import pytest

import examples.social_graph_bot as sg


class DummySession:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, *args, **kwargs):
        raise aiohttp.ClientError("boom")


@pytest.mark.asyncio
async def test_send_to_prism_client_error(monkeypatch, caplog):
    monkeypatch.setattr(sg.aiohttp, "ClientSession", lambda: DummySession())
    with caplog.at_level(logging.WARNING):
        await sg.send_to_prism({"x": 1})
    assert any("ClientError" in r.getMessage() for r in caplog.records)
