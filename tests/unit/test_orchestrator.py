import asyncio
from types import SimpleNamespace

import pytest

import deepthought.orchestrator as orchestrator


class DummyService:
    def __init__(self, nc, js):
        self.started = False
        self.stopped = False

    async def start(self, durable_name: str = "d") -> bool:
        self.started = True
        return True

    async def stop(self) -> None:
        self.stopped = True


@pytest.mark.asyncio
async def test_run(monkeypatch, tmp_path):
    created: dict[str, DummyService] = {}

    orig_init = DummyService.__init__

    def capture_init(self, nc, js):
        orig_init(self, nc, js)
        created["instance"] = self

    monkeypatch.setattr(DummyService, "__init__", capture_init)

    ep = SimpleNamespace(name="dummy", load=lambda: DummyService)
    monkeypatch.setattr(
        orchestrator.metadata,
        "entry_points",
        lambda: SimpleNamespace(select=lambda **k: [ep]),
    )

    class DummyNATS:
        def __init__(self) -> None:
            self.is_connected = True

        async def drain(self):
            pass

    class DummyJS:
        pass

    async def fake_connect():
        return DummyNATS(), DummyJS()

    monkeypatch.setattr(orchestrator, "_connect_nats", fake_connect)

    monkeypatch.setattr(asyncio.Event, "wait", lambda self: asyncio.sleep(0))

    cfg = tmp_path / "cfg.json"
    cfg.write_text('{"services": ["dummy"]}', encoding="utf-8")

    await orchestrator.run(str(cfg))

    service = created.get("instance")
    assert service is not None, "DummyService instance was not created"
    assert service.started is True
    assert service.stopped is True
