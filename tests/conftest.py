# Standard library imports
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

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
if "deepthought.motivate" not in sys.modules:
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
