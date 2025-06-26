import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Provide a lightweight stub for sentence_transformers if the package is missing.

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
