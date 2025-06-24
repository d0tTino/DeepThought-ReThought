import importlib

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
