import importlib
import sys
import types

import pytest

import examples.social_graph_bot as sg


@pytest.mark.parametrize("backend", [None, "vader"])
def test_sentiment_backend(monkeypatch, backend):
    if backend is None:
        monkeypatch.delenv("SENTIMENT_BACKEND", raising=False)
    else:
        monkeypatch.setenv("SENTIMENT_BACKEND", backend)
    module = importlib.reload(sg)

    score_pos = module.analyze_sentiment("I love this")
    score_neg = module.analyze_sentiment("I hate this")
    assert score_pos > score_neg

    # restore default for other tests
    monkeypatch.delenv("SENTIMENT_BACKEND", raising=False)
    importlib.reload(sg)


def test_invalid_backend_defaults_to_textblob(monkeypatch):
    monkeypatch.setenv("SENTIMENT_BACKEND", "unknown")

    class Dummy:
        def __init__(self, text: str) -> None:
            self.text = text

        @property
        def sentiment(self):
            return types.SimpleNamespace(polarity=0.42)

    original_tb = sys.modules.get("textblob")
    monkeypatch.setitem(sys.modules, "textblob", types.SimpleNamespace(TextBlob=Dummy))
    module = importlib.reload(sg)
    assert module.analyze_sentiment("hello") == 0.42

    monkeypatch.delenv("SENTIMENT_BACKEND", raising=False)
    if original_tb is not None:
        monkeypatch.setitem(sys.modules, "textblob", original_tb)
    else:
        monkeypatch.delitem(sys.modules, "textblob", raising=False)
    importlib.reload(sg)
