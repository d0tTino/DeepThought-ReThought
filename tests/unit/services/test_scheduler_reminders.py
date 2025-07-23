import asyncio
import json
import sys
import types
from datetime import datetime, timedelta, timezone

import pytest

# stub nats and related dependencies
fake_nats = types.ModuleType("nats")
fake_nats.aio = types.ModuleType("aio")
fake_client_mod = types.ModuleType("client")
setattr(fake_client_mod, "Client", object)
fake_nats.aio.client = fake_client_mod
fake_msg_mod = types.ModuleType("msg")
setattr(fake_msg_mod, "Msg", object)
fake_nats.aio.msg = fake_msg_mod
fake_nats.js = types.ModuleType("js")
fake_js_client_mod = types.ModuleType("client")
setattr(fake_js_client_mod, "JetStreamContext", object)
fake_nats.js.client = fake_js_client_mod
fake_errors_mod = types.ModuleType("errors")
setattr(fake_errors_mod, "Error", Exception)
fake_nats.errors = fake_errors_mod
sys.modules.setdefault("nats", fake_nats)
sys.modules.setdefault("nats.aio", fake_nats.aio)
sys.modules.setdefault("nats.aio.client", fake_client_mod)
sys.modules.setdefault("nats.aio.msg", fake_msg_mod)
sys.modules.setdefault("nats.js", fake_nats.js)
sys.modules.setdefault("nats.js.client", fake_js_client_mod)
sys.modules.setdefault("nats.errors", fake_errors_mod)
sys.modules.setdefault("aiosqlite", types.ModuleType("aiosqlite"))
fake_pyd = types.ModuleType("pydantic")
fake_pyd.AnyUrl = str
fake_pyd.ValidationError = Exception
fake_pyd.Field = lambda default=None, **kw: default
sys.modules.setdefault("pydantic", fake_pyd)
fake_ps = types.ModuleType("pydantic_settings")


class DummyBase:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


fake_ps.BaseSettings = DummyBase
fake_ps.SettingsConfigDict = dict
sys.modules.setdefault("pydantic_settings", fake_ps)
fake_nx = types.ModuleType("networkx")
setattr(fake_nx, "DiGraph", object)
import importlib.machinery
fake_nx.__spec__ = importlib.machinery.ModuleSpec("networkx", loader=None)
fake_prom = types.ModuleType("prometheus_client")


class _Metric:
    def labels(self, **kwargs):
        return self

    def inc(self, *args, **kwargs):
        pass

    def observe(self, *args, **kwargs):
        pass


fake_prom.Counter = lambda *a, **k: _Metric()
fake_prom.Histogram = lambda *a, **k: _Metric()
fake_prom.REGISTRY = types.SimpleNamespace(_names_to_collectors={})
sys.modules.setdefault("prometheus_client", fake_prom)
sys.modules.setdefault("networkx", fake_nx)

from deepthought.eda.events import EventSubjects
from deepthought.services.scheduler import SchedulerService


class DummyPublisher:
    def __init__(self):
        self.published = []

    async def publish(self, subject, payload, use_jetstream=True, timeout=10.0):
        self.published.append((subject, payload))


class DummyMemoryDAL:
    def __init__(self, interactions):
        self.interactions = interactions

    def get_recent_facts(self, count=3):
        return self.interactions[-count:]


class DummyGraphDAL:
    def __init__(self):
        self.entities = []

    def add_entity(self, label, props):
        self.entities.append((label, props))


class DummySummaryDB:
    def __init__(self, rows):
        self.rows = rows
        self.marked = []

    async def add_summary_goal(self, *a, **kw):
        pass

    async def list_pending_summary_goals(self):
        return self.rows

    async def mark_summary_goal_done(self, task_id):
        self.marked.append(task_id)


@pytest.mark.asyncio
async def test_scheduler_multiple_reminders(monkeypatch):
    module = sys.modules.setdefault("examples.social_graph_bot", types.ModuleType("examples.social_graph_bot"))
    module.generate_reflection = lambda _t: "ok"
    current = datetime(2024, 1, 1, tzinfo=timezone.utc)

    def now():
        return current

    real_sleep = asyncio.sleep

    async def fake_sleep(seconds):
        nonlocal current
        current += timedelta(seconds=seconds)
        await real_sleep(0)

    publisher = DummyPublisher()
    memory = DummyMemoryDAL([])
    graph = DummyGraphDAL()

    service = SchedulerService(
        publisher,
        memory,
        graph,
        summary_interval=100.0,
        now_func=now,
        sleep_func=fake_sleep,
    )

    await service.start()
    await fake_sleep(0)
    service.schedule_reminder("first", now() + timedelta(seconds=2), "r1")
    service.schedule_reminder("second", now() + timedelta(seconds=4), "r2")

    await fake_sleep(5)
    await service.stop()

    messages = [p.message for _s, p in publisher.published]
    assert messages == ["first", "second"]


@pytest.mark.asyncio
async def test_goal_loop_skips_bad_rows(monkeypatch):
    module = sys.modules.setdefault("examples.social_graph_bot", types.ModuleType("examples.social_graph_bot"))
    module.generate_reflection = lambda _t: "ok"
    current = datetime(2024, 1, 1, tzinfo=timezone.utc)

    def now():
        return current

    async def fake_sleep(seconds):
        nonlocal current
        current += timedelta(seconds=seconds)

    publisher = DummyPublisher()
    memory = DummyMemoryDAL([])
    graph = DummyGraphDAL()

    rows = [
        (1, 0, "not json", "p1"),
        (2, 0, json.dumps({"due": now().isoformat()}), "p2"),
        (3, 0, json.dumps({"goal": "g"}), "p3"),
        (4, 0, json.dumps({"due": "bad", "goal": "g"}), "p4"),
        (5, 0, json.dumps({"due": now().isoformat(), "goal": "good"}), "p5"),
    ]
    summary_db = DummySummaryDB(rows)

    service = SchedulerService(
        publisher,
        memory,
        graph,
        summary_interval=100.0,
        now_func=now,
        sleep_func=fake_sleep,
        summary_db=summary_db,
    )

    async def list_once():
        service._running = False
        return summary_db.rows

    summary_db.list_pending_summary_goals = list_once
    service._running = True
    await service._goal_loop()

    assert summary_db.marked == [5]
    assert len(publisher.published) == 1
    subj, payload = publisher.published[0]
    assert subj == EventSubjects.REMINDER_TRIGGERED
    assert payload.message == "good"
