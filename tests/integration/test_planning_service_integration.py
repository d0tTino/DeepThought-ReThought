import asyncio
import json
import os
import sys
import types

import pytest

# Stub minimal pydantic classes before importing deepthought modules
fake_pyd = types.ModuleType("pydantic")
fake_pyd.AnyUrl = str
fake_pyd.ValidationError = Exception
fake_pyd.Field = lambda default=None, **kwargs: default
sys.modules.setdefault("pydantic", fake_pyd)

fake_ps = types.ModuleType("pydantic_settings")
fake_ps.BaseSettings = object
fake_ps.SettingsConfigDict = dict
sys.modules.setdefault("pydantic_settings", fake_ps)

# Stub aiosqlite used by DBManager to avoid optional dependency
fake_aiosqlite = types.ModuleType("aiosqlite")
fake_aiosqlite.Connection = object
fake_aiosqlite.connect = lambda *a, **k: types.SimpleNamespace(
    execute=lambda *a, **k: None,
    commit=lambda: None,
    close=lambda: None,
)
sys.modules.setdefault("aiosqlite", fake_aiosqlite)

# Stub textblob used by DBManager sentiment analysis
fake_textblob = types.ModuleType("textblob")
fake_textblob.TextBlob = lambda text: types.SimpleNamespace(sentiment=types.SimpleNamespace(polarity=0.0))
sys.modules.setdefault("textblob", fake_textblob)

# Stub rdflib used by reasoning_service imports
fake_rdflib = types.ModuleType("rdflib")
fake_rdflib.Namespace = object
fake_rdflib.Graph = object
fake_rdflib.URIRef = str
fake_rdflib.namespace = types.SimpleNamespace(RDF=object())
sys.modules.setdefault("rdflib", fake_rdflib)
sys.modules.setdefault("rdflib.namespace", fake_rdflib.namespace)

# Stub prometheus_client used by deepthought.metrics
fake_prom = types.ModuleType("prometheus_client")
fake_prom.Counter = lambda *a, **k: object()
fake_prom.Histogram = lambda *a, **k: object()
fake_prom.REGISTRY = types.SimpleNamespace(_names_to_collectors={})
sys.modules.setdefault("prometheus_client", fake_prom)

# Provide a minimal pyperplan stub to avoid heavy dependency
fake_pyperplan = types.ModuleType("pyperplan")
pddl_mod = types.ModuleType("pyperplan.pddl")
parser_mod = types.ModuleType("pyperplan.pddl.parser")
parser_mod.Parser = object
pddl_mod.parser = parser_mod
fake_pyperplan.pddl = pddl_mod
fake_pyperplan.planner = types.SimpleNamespace(_ground=lambda *a, **k: None)
fake_pyperplan.search = types.SimpleNamespace(breadth_first_search=lambda *a, **k: [])
sys.modules.setdefault("pyperplan", fake_pyperplan)
sys.modules.setdefault("pyperplan.pddl", pddl_mod)
sys.modules.setdefault("pyperplan.pddl.parser", parser_mod)
sys.modules.setdefault("pyperplan.planner", fake_pyperplan.planner)
sys.modules.setdefault("pyperplan.search", fake_pyperplan.search)

pytest.importorskip("nats")

from nats.aio.client import Client as NATS
from nats.js.api import DiscardPolicy, RetentionPolicy, StorageType, StreamConfig

from deepthought.eda.events import EventSubjects, PlanRequestedPayload
from deepthought.services.planning_service import PlanningService
from tests.helpers import nats_server_available
from tests.integration.test_orchestrator_full import (
    ensure_stream_exists,
    get_nats_url,
    nats_container,
)

pytestmark = pytest.mark.nats

STREAM_NAME = "deepthought_events"


async def _ensure_chat_stream(js):
    try:
        info = await js.stream_info(STREAM_NAME)
        if "chat.raw" not in info.config.subjects:
            cfg = StreamConfig(
                name=info.config.name,
                subjects=list(info.config.subjects) + ["chat.raw"],
                retention=info.config.retention,
                storage=info.config.storage,
                discard=info.config.discard,
                max_msgs_per_subject=info.config.max_msgs_per_subject,
            )
            await js.update_stream(cfg)
    except Exception:
        cfg = StreamConfig(
            name=STREAM_NAME,
            subjects=["dtr.>", "chat.raw"],
            retention=RetentionPolicy.LIMITS,
            storage=StorageType.MEMORY,
            max_msgs_per_subject=100,
            discard=DiscardPolicy.OLD,
        )
        await js.add_stream(cfg)


@pytest.mark.asyncio
async def test_planning_service_generates_actions(monkeypatch, tmp_path, nats_container):
    if not nats_server_available(get_nats_url()):
        pytest.skip("NATS server not available")

    class DummyTranslator:
        def translate(self, goal: str):
            return "d", "p"

    monkeypatch.setattr("deepthought.services.planning_service.L2PTranslator", DummyTranslator)
    monkeypatch.setattr("deepthought.services.planning_service.plan", lambda d, p: ["a1", "a2"])

    nc = await NATS().connect(servers=[get_nats_url()], connect_timeout=10)
    js = nc.jetstream(timeout=10.0)
    await ensure_stream_exists(js)
    await _ensure_chat_stream(js)

    service = PlanningService(nc, js)
    await service.start(durable_name="plan_test")

    received = []
    done = asyncio.Event()

    async def handler(msg):
        received.append(msg.data.decode())
        if len(received) >= 2:
            done.set()
        await msg.ack()

    sub = await js.subscribe(
        subject=EventSubjects.CHAT_RAW,
        durable="sink",
        cb=handler,
        stream=STREAM_NAME,
    )

    payload = PlanRequestedPayload(goal="move obj from a to b", input_id="1")
    await js.publish(EventSubjects.PLAN_REQUESTED, payload.to_json().encode())

    await asyncio.wait_for(done.wait(), timeout=10.0)
    assert received == ["a1", "a2"]

    await sub.unsubscribe()
    await service.stop()
    await nc.drain()
