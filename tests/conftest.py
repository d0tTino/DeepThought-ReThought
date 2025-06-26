# Standard library imports
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Provide a lightweight stub of the social_graph_bot module so tests do not
# require optional heavy dependencies from the full example implementation.
sg_stub = types.ModuleType("examples.social_graph_bot")


async def _noop(*args, **kwargs):
    return None


sg_stub.send_to_prism = _noop
sg_stub.publish_input_received = _noop
sys.modules.setdefault("examples.social_graph_bot", sg_stub)

# Stub out optional dependencies so tests can run without installing them.
if "nats" not in sys.modules:
    nats_stub = types.ModuleType("nats")
    nats_stub.errors = types.SimpleNamespace(Error=Exception, TimeoutError=Exception)
    sys.modules["nats"] = nats_stub
    aio_mod = types.ModuleType("nats.aio")
    client_mod = types.ModuleType("nats.aio.client")
    msg_mod = types.ModuleType("nats.aio.msg")

    class DummyClient:
        pass

    class DummyMsg:
        pass

    client_mod.Client = DummyClient
    msg_mod.Msg = DummyMsg
    aio_mod.client = client_mod
    sys.modules["nats.aio"] = aio_mod
    sys.modules["nats.aio.client"] = client_mod
    sys.modules["nats.aio.msg"] = msg_mod

    js_mod = types.ModuleType("nats.js")
    client_js_mod = types.ModuleType("nats.js.client")
    api_mod = types.ModuleType("nats.js.api")

    class DummyJetStreamContext:
        pass

    client_js_mod.JetStreamContext = DummyJetStreamContext
    js_mod.client = client_js_mod
    js_mod.JetStreamContext = DummyJetStreamContext
    api_mod.DiscardPolicy = object
    api_mod.RetentionPolicy = object
    api_mod.StorageType = object
    api_mod.StreamConfig = object
    sys.modules["nats.js"] = js_mod
    sys.modules["nats.js.client"] = client_js_mod
    sys.modules["nats.js.api"] = api_mod

if "aiosqlite" not in sys.modules:
    aiosqlite_stub = types.ModuleType("aiosqlite")

    async def connect(*args, **kwargs):
        raise RuntimeError("aiosqlite stub used")

    aiosqlite_stub.connect = connect
    sys.modules["aiosqlite"] = aiosqlite_stub

# Provide a stub for ``sentence_transformers`` if the package is missing so
# that RewardManager can be imported without heavy dependencies.

if "sentence_transformers" not in sys.modules:
    st = types.ModuleType("sentence_transformers")

    class DummyModel:
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, text, convert_to_numpy=True):
            import numpy as np
            return np.array([len(text)], dtype=float)

    st.SentenceTransformer = DummyModel
    st.util = types.SimpleNamespace(cos_sim=lambda a, b: [[0.0]])
    sys.modules["sentence_transformers"] = st

    sys.modules["sentence_transformers.util"] = st.util


# Provide a minimal stub for ``send_to_prism`` and ``publish_input_received`` on
# the social_graph_bot module so tests can intercept these calls. The stub must
# be applied after ensuring ``sentence_transformers`` is available so the module
# imports cleanly.
async def _noop(*args, **kwargs):
    return None

import examples.social_graph_bot as sg

@pytest.fixture
def prism_calls(monkeypatch):
    calls = []

    async def fake_send(data):
        calls.append(data)

    monkeypatch.setattr(sg, "send_to_prism", fake_send)
    return calls


@pytest.fixture
def input_events(monkeypatch):
    calls = []

    async def fake_publish(text):
        calls.append(text)

    monkeypatch.setattr(sg, "publish_input_received", fake_publish)
    return calls
