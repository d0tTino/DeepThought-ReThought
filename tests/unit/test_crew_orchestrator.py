import asyncio
import importlib

import pytest


def _load_orchestrator_module():
    return importlib.import_module("deepthought.orchestrator")


class DummyCrew:
    def __init__(self):
        self.started = False
        self.stopped = False

    async def __aenter__(self):
        self.started = True
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.stopped = True


@pytest.mark.asyncio
async def test_run_with_crew(monkeypatch, tmp_path):
    orchestrator = _load_orchestrator_module()
    created = {}

    def factory():
        crew = DummyCrew()
        created["crew"] = crew
        return crew

    class DummyNATS:
        is_connected = True

        async def drain(self):
            pass

    async def fake_connect():
        return DummyNATS(), object()

    monkeypatch.setattr(orchestrator, "_connect_nats", fake_connect)

    monkeypatch.setattr(orchestrator, "discover_services", lambda names: [])
    monkeypatch.setattr(orchestrator, "_load_callable", lambda path: factory)
    monkeypatch.setattr(asyncio.Event, "wait", lambda self: asyncio.sleep(0))

    cfg = tmp_path / "c.yaml"
    cfg.write_text("crews:\n  - dummy:factory\n", encoding="utf-8")

    await orchestrator.run(str(cfg))

    crew = created.get("crew")
    assert crew is not None and crew.started and crew.stopped
