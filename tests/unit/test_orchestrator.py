import asyncio
import importlib
import sys
from types import ModuleType, SimpleNamespace

fake_nats = ModuleType("nats")
fake_nats.__spec__ = importlib.machinery.ModuleSpec("nats", loader=None)
fake_nats.aio = ModuleType("aio")
fake_client_mod = ModuleType("client")
setattr(fake_client_mod, "Client", object)
fake_nats.aio.client = fake_client_mod
fake_msg_mod = ModuleType("msg")
setattr(fake_msg_mod, "Msg", object)
fake_nats.aio.msg = fake_msg_mod
fake_nats.js = ModuleType("js")
fake_js_client_mod = ModuleType("client")
setattr(fake_js_client_mod, "JetStreamContext", object)
fake_nats.js.client = fake_js_client_mod
fake_errors_mod = ModuleType("errors")
setattr(fake_errors_mod, "Error", Exception)
fake_nats.errors = fake_errors_mod
sys.modules.setdefault("nats", fake_nats)
sys.modules.setdefault("nats.aio", fake_nats.aio)
sys.modules.setdefault("nats.aio.client", fake_client_mod)
sys.modules.setdefault("nats.aio.msg", fake_msg_mod)
sys.modules.setdefault("nats.js", fake_nats.js)
sys.modules.setdefault("nats.js.client", fake_js_client_mod)
sys.modules.setdefault("nats.errors", fake_errors_mod)

import pytest

from deepthought.eda.events import EventSubjects


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

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.stop()


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

    class DummySub:
        def __init__(self, *a, **k):
            pass

        async def subscribe(self, *a, **k):
            return True

        async def unsubscribe_all(self):
            pass

    class DummyPub:
        def __init__(self, *a, **k):
            pass

        async def publish(self, *a, **k):
            return True

    monkeypatch.setattr(orchestrator, "Subscriber", DummySub)
    monkeypatch.setattr(orchestrator, "Publisher", DummyPub)

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


@pytest.mark.asyncio
async def test_plan_subscription(monkeypatch, tmp_path):
    orchestrator = _load_orchestrator_module()
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

    recorded = {}

    class DummySub:
        def __init__(self, *a, **k):
            recorded["sub"] = []

        async def subscribe(self, subject, handler, **kw):
            recorded["sub"].append(subject)
            return True

        async def unsubscribe_all(self):
            pass

    class DummyPub:
        def __init__(self, *a, **k):
            pass

        async def publish(self, *a, **k):
            recorded["published"] = True

    monkeypatch.setattr(orchestrator, "Publisher", DummyPub)
    monkeypatch.setattr(orchestrator, "Subscriber", DummySub)

    class DummyTranslator:
        def translate(self, goal):
            return "d", "p"

    class DummyPlanner:
        @staticmethod
        def plan(d, p):
            return ["ok"]

    monkeypatch.setattr(orchestrator.translator, "L2PTranslator", DummyTranslator)
    monkeypatch.setattr(orchestrator.planner, "plan", DummyPlanner.plan)

    monkeypatch.setattr(asyncio.Event, "wait", lambda self: asyncio.sleep(0))

    cfg = tmp_path / "c.json"
    cfg.write_text('{"services": ["dummy"]}', encoding="utf-8")

    await orchestrator.run(str(cfg))

    assert EventSubjects.PLAN_REQUESTED in recorded.get("sub", [])
