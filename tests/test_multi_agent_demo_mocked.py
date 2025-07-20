import asyncio
import importlib
import sys
import types

import pytest

sys.modules.pop("examples.multi_agent_demo", None)
sys.modules.pop("langgraph", None)

# Provide a very small stub of langgraph so the demo can run without the real
# package installed.
fake_langgraph = types.ModuleType("langgraph")
fake_graph = types.ModuleType("langgraph.graph")


class StateGraph:
    def __init__(self, _state):
        self.nodes = {}
        self.entry = None

    def add_node(self, name, fn):
        self.nodes[name] = fn

    def set_entry_point(self, name):
        self.entry = name

    def add_edge(self, *_args):
        pass

    def compile(self):
        graph = self

        class Compiled:
            async def ainvoke(self, state):
                fn = graph.nodes[graph.entry]
                state = await fn(state) if asyncio.iscoroutinefunction(fn) else fn(state)
                for name, fn in graph.nodes.items():
                    if name != graph.entry:
                        state = await fn(state) if asyncio.iscoroutinefunction(fn) else fn(state)
                return state

        return Compiled()


fake_graph.StateGraph = StateGraph
fake_langgraph.graph = fake_graph
sys.modules.setdefault("langgraph", fake_langgraph)
sys.modules.setdefault("langgraph.graph", fake_graph)

pytest.importorskip("langgraph.graph")


class DummyMsg:
    def __init__(self, subject: str, data: bytes):
        self.subject = subject
        self.data = data
        self.acked = False
        self.nacked = False

    async def ack(self):
        self.acked = True

    async def nak(self):
        self.nacked = True


class DummyBus:
    def __init__(self):
        self.subscribers: dict[str, list] = {}

    def subscribe(self, subject: str, cb):
        self.subscribers.setdefault(subject, []).append(cb)
        return DummySubscription(self, subject, cb)

    def unsubscribe(self, subject: str, cb):
        if subject in self.subscribers:
            self.subscribers[subject].remove(cb)

    async def publish(self, subject: str, data: bytes):
        for cb in list(self.subscribers.get(subject, [])):
            await cb(DummyMsg(subject, data))


class DummySubscription:
    def __init__(self, bus: DummyBus, subject: str, cb):
        self._bus = bus
        self._subject = subject
        self._cb = cb

    async def unsubscribe(self):
        self._bus.unsubscribe(self._subject, self._cb)


class DummyJS:
    def __init__(self, bus: DummyBus):
        self._bus = bus

    async def publish(self, subject, data, timeout=10.0):
        await self._bus.publish(subject, data)
        return types.SimpleNamespace(seq=1, stream="s")

    async def subscribe(self, subject, *, queue="", durable="", cb=None, manual_ack=False):
        return self._bus.subscribe(subject, cb)

    async def add_stream(self, config):
        return None

    async def stream_info(self, name):
        raise Exception("missing")


class DummyNATS:
    def __init__(self, bus: DummyBus):
        self.is_connected = True
        self._bus = bus

    def jetstream(self):
        return DummyJS(self._bus)

    async def drain(self):
        return None


bus = DummyBus()


async def dummy_connect(*args, **kwargs):
    return DummyNATS(bus)


# Create fake nats module hierarchy
fake_nats = types.ModuleType("nats")
fake_nats.__spec__ = importlib.machinery.ModuleSpec("nats", loader=None)
fake_nats.connect = dummy_connect
fake_nats.errors = types.SimpleNamespace(Error=Exception, TimeoutError=Exception)
fake_nats.aio = types.ModuleType("aio")
fake_nats.aio.client = types.ModuleType("client")
fake_nats.aio.client.Client = object
fake_nats.aio.msg = types.ModuleType("msg")
fake_nats.aio.msg.Msg = object
fake_nats.js = types.ModuleType("js")
fake_nats.js.client = types.ModuleType("client")
fake_nats.js.client.JetStreamContext = object
fake_nats.js.api = types.ModuleType("api")


class DiscardPolicy:
    OLD = "old"


class RetentionPolicy:
    LIMITS = "limits"


class StorageType:
    MEMORY = "memory"


class StreamConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


fake_nats.js.api.DiscardPolicy = DiscardPolicy
fake_nats.js.api.RetentionPolicy = RetentionPolicy
fake_nats.js.api.StorageType = StorageType
fake_nats.js.api.StreamConfig = StreamConfig

sys.modules.setdefault("nats", fake_nats)
sys.modules.setdefault("nats.aio", fake_nats.aio)
sys.modules.setdefault("nats.aio.client", fake_nats.aio.client)
sys.modules.setdefault("nats.aio.msg", fake_nats.aio.msg)
sys.modules.setdefault("nats.js", fake_nats.js)
sys.modules.setdefault("nats.js.client", fake_nats.js.client)
sys.modules.setdefault("nats.js.api", fake_nats.js.api)
sys.modules.setdefault("nats.errors", fake_nats.errors)

fake_prom = types.ModuleType("prometheus_client")


class _Metric:
    def labels(self, *args, **kwargs):
        return self

    def inc(self, *args, **kwargs):
        pass

    def observe(self, *args, **kwargs):
        pass


fake_prom.Counter = lambda *a, **k: _Metric()
fake_prom.Histogram = lambda *a, **k: _Metric()
fake_prom.start_http_server = lambda *a, **k: None
fake_prom.REGISTRY = types.SimpleNamespace(_names_to_collectors={})
sys.modules.setdefault("prometheus_client", fake_prom)


import examples.multi_agent_demo as demo


@pytest.mark.asyncio
async def test_multi_agent_demo_main(monkeypatch):
    async def fake_generate(self, prompt: str) -> str:
        return f"echo: {prompt}"

    monkeypatch.setattr(demo, "nats", fake_nats)
    monkeypatch.setattr(demo.RemoteLLM, "_generate", fake_generate)
    orig_sleep = asyncio.sleep
    monkeypatch.setattr(demo.asyncio, "sleep", lambda *a, **k: orig_sleep(0))
    monkeypatch.setenv("LANGGRAPH_RECURSION_LIMIT", "1000")

    await demo.main()

    for handler in demo.output_handlers:
        assert handler.get_all_responses(), "handler received no messages"


@pytest.mark.asyncio
async def test_no_metrics_server_when_port_zero(monkeypatch):
    calls: list[int] = []

    async def fake_generate(self, prompt: str) -> str:
        return f"echo: {prompt}"

    monkeypatch.setattr(demo, "nats", fake_nats)
    monkeypatch.setattr(demo.RemoteLLM, "_generate", fake_generate)
    monkeypatch.setattr(demo, "start_http_server", lambda port: calls.append(port))
    orig_sleep = asyncio.sleep
    monkeypatch.setattr(demo.asyncio, "sleep", lambda *a, **k: orig_sleep(0))
    monkeypatch.setenv("LANGGRAPH_RECURSION_LIMIT", "1000")
    monkeypatch.setenv("METRICS_PORT", "0")

    await demo.main()

    assert not calls, "start_http_server should not be called when METRICS_PORT=0"
