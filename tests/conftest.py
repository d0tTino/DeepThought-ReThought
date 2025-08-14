# Standard library imports
import os
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Avoid importing heavy optional dependencies during test collection.
os.environ.setdefault("DEEPTHOUGHT_LIGHT_IMPORT", "1")

# Ensure the real prometheus_client package is loaded before tests
# possibly insert a stub using sys.modules.setdefault.
try:  # pragma: no cover - optional dependency may be missing
    import prometheus_client  # noqa: F401
except Exception:
    pass

# Ensure the real networkx package is loaded before tests may
# insert a stub using ``sys.modules.setdefault``.
try:  # pragma: no cover - optional dependency may be missing
    import networkx  # noqa: F401
except Exception:
    pass

# Provide a stub for the ``nats`` package when it is unavailable so modules
# importing it (e.g. the event publisher) can be loaded during tests.
try:  # pragma: no cover - optional dependency may be missing
    import importlib.util

    spec = importlib.util.find_spec("nats")
except Exception:
    spec = None

if spec is None:
    nats_stub = types.ModuleType("nats")
    nats_stub.aio = types.ModuleType("aio")
    aio_client_mod = types.ModuleType("client")
    setattr(aio_client_mod, "Client", object)
    nats_stub.aio.client = aio_client_mod
    nats_stub.js = types.ModuleType("js")
    js_client_mod = types.ModuleType("client")
    setattr(js_client_mod, "JetStreamContext", object)
    nats_stub.js.client = js_client_mod
    api_mod = types.ModuleType("api")

    class _StreamConfig:
        def __init__(self, *args, **kwargs) -> None:
            pass

    class _DiscardPolicy:
        Old = 0

    class _RetentionPolicy:
        LIMITS = 0

    class _StorageType:
        MEMORY = 0

    api_mod.StreamConfig = _StreamConfig
    api_mod.DiscardPolicy = _DiscardPolicy
    api_mod.RetentionPolicy = _RetentionPolicy
    api_mod.StorageType = _StorageType
    nats_stub.js.api = api_mod
    msg_mod = types.ModuleType("msg")
    setattr(msg_mod, "Msg", object)
    nats_stub.aio.msg = msg_mod
    nats_stub.errors = types.ModuleType("errors")
    sys.modules.setdefault("nats", nats_stub)
    sys.modules.setdefault("nats.aio", nats_stub.aio)
    sys.modules.setdefault("nats.aio.client", nats_stub.aio.client)
    sys.modules.setdefault("nats.aio.msg", nats_stub.aio.msg)
    sys.modules.setdefault("nats.js", nats_stub.js)
    sys.modules.setdefault("nats.js.client", nats_stub.js.client)
    sys.modules.setdefault("nats.js.api", nats_stub.js.api)
    sys.modules.setdefault("nats.errors", nats_stub.errors)

# Provide lightweight stubs for optional heavy packages so tests can run in
# minimal environments.
if "torch" not in sys.modules:
    try:  # pragma: no cover - best effort to import the real library
        import torch  # noqa: F401
    except Exception:
        torch_stub = types.ModuleType("torch")


        class _NoGrad:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        torch_stub.no_grad = lambda: _NoGrad()
        torch_stub.softmax = lambda x, dim=None: x
        sys.modules["torch"] = torch_stub

if "l2p" not in sys.modules:
    l2p_stub = types.ModuleType("l2p")

    def _parse_domain(*args, **kwargs):
        return {}

    def _parse_problem(*args, **kwargs):
        return {}

    l2p_stub.utils = types.SimpleNamespace(parse_domain=_parse_domain, parse_problem=_parse_problem)
    sys.modules["l2p"] = l2p_stub
    sys.modules["l2p.utils"] = l2p_stub.utils

if "owlready2" not in sys.modules:
    owlready_stub = types.ModuleType("owlready2")

    class ThingClass:  # type: ignore
        pass

    class World:  # type: ignore
        pass

    def sync_reasoner_hermit(*args, **kwargs):
        return None

    owlready_stub.ThingClass = ThingClass
    owlready_stub.World = World
    owlready_stub.sync_reasoner_hermit = sync_reasoner_hermit
    sys.modules["owlready2"] = owlready_stub

if "deepthought.motivate.ledger" not in sys.modules:
    ledger_stub = types.ModuleType("ledger")

    class Ledger:  # type: ignore
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def add_event(self, *args, **kwargs) -> None:
            return None

    ledger_stub.Ledger = Ledger
    sys.modules.setdefault("deepthought.motivate.ledger", ledger_stub)

if "transformers" not in sys.modules:
    transformers_stub = types.ModuleType("transformers")

    class _DummyModel:
        def __init__(self, *args, **kwargs) -> None:
            pass

    transformers_stub.AutoModelForSequenceClassification = _DummyModel
    transformers_stub.AutoTokenizer = _DummyModel
    sys.modules["transformers"] = transformers_stub

import pytest

# Provide a lightweight stub of the social_graph_bot module. This allows tests
# to run without installing optional heavy dependencies used by the full
# example implementation.

sg_stub = types.ModuleType("examples.social_graph_bot")


async def _noop(*args, **kwargs):
    return None


sg_stub.send_to_prism = _noop
sg_stub.publish_input_received = _noop

try:  # Attempt to load the real example module
    import importlib

    sg = importlib.import_module("examples.social_graph_bot")
    sg.send_to_prism = _noop
except Exception:  # pragma: no cover - fallback when dependencies are missing
    sg = sg_stub
sys.modules["examples.social_graph_bot"] = sg


# Provide a lightweight stub for sentence_transformers if the package is
# missing so that modules importing RewardManager can be loaded without the
# heavy optional dependency.

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

# Provide a lightweight fallback for the ``deepthought.motivate`` package if it
# isn't installed. Several tests insert a dummy module using
# ``sys.modules.setdefault`` which can break imports that expect the real
# submodules. Registering this stub early ensures those imports succeed even when
# the optional package is missing.
import importlib.util

if importlib.util.find_spec("deepthought.motivate") is None:
    motivate = types.ModuleType("motivate")

    caption = types.ModuleType("caption")

    def summarise_message(message: str, max_words: int = 5) -> str:
        return " ".join(message.split()[:max_words])

    caption.summarise_message = summarise_message

    scorer = types.ModuleType("scorer")

    def score_caption(caption_str: str, nonce: str) -> int:
        from hashlib import sha256

        digest = sha256((nonce + caption_str).encode()).digest()
        return 1 + digest[0] % 7

    scorer.score_caption = score_caption

    motivate.caption = caption
    motivate.scorer = scorer
    sys.modules["deepthought.motivate"] = motivate
    sys.modules["deepthought.motivate.caption"] = caption
    sys.modules["deepthought.motivate.scorer"] = scorer


# Provide a minimal stub for ``send_to_prism`` and ``publish_input_received`` on
# the social_graph_bot module so tests can intercept these calls. The stub must
# be applied after ensuring ``sentence_transformers`` is available so the module
# imports cleanly.


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


def pytest_collection_modifyitems(config, items):
    import importlib.util

    try:
        spec = importlib.util.find_spec("nats")
    except Exception:
        spec = None

    if spec is not None:
        return
    skip = pytest.mark.skip(reason="nats not installed")
    for item in items:
        if "nats" in item.keywords:
            item.add_marker(skip)
