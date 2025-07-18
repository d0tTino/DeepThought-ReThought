import asyncio
import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest


def _load_orchestrator_module():
    if "pydantic" not in sys.modules:
        pydantic_mod = ModuleType("pydantic")

        class AnyUrl(str):
            pass

        def Field(default=None, **k):
            return default

        class ValidationError(Exception):
            pass

        pydantic_mod.AnyUrl = AnyUrl
        pydantic_mod.Field = Field
        pydantic_mod.ValidationError = ValidationError
        sys.modules["pydantic"] = pydantic_mod

    if "pydantic_settings" not in sys.modules:
        ps_mod = ModuleType("pydantic_settings")

        class BaseSettings:
            model_config: dict = {}

        ps_mod.BaseSettings = BaseSettings
        ps_mod.SettingsConfigDict = dict
        sys.modules["pydantic_settings"] = ps_mod
    return importlib.import_module("deepthought.orchestrator")


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
    orchestrator = _load_orchestrator_module()
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


def test_discover_services_fallback(monkeypatch):
    orchestrator = _load_orchestrator_module()
    ep = SimpleNamespace(name="dummy", load=lambda: DummyService)

    def dummy_entry_points():
        return {"deepthought.services": [ep]}

    monkeypatch.setattr(orchestrator.metadata, "entry_points", dummy_entry_points)

    services = orchestrator.discover_services(["dummy"])
    assert services == [DummyService]
