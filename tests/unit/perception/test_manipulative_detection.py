import importlib

import pytest

from deepthought.perception import manipulative_detection as md
from deepthought.perception.manipulative_phrases import (
    DEFLECTION_PHRASES,
    FLATTERY_PHRASES,
    GASLIGHTING_PHRASES,
    GUILT_TRIP_PHRASES,
    THREAT_PHRASES,
)


@pytest.mark.parametrize("phrase", GUILT_TRIP_PHRASES)
def test_guilt_trip_phrases(phrase):
    assert md.detect_manipulation(phrase) == "guilt_tripping"


@pytest.mark.parametrize("phrase", THREAT_PHRASES)
def test_threat_phrases(phrase):
    assert md.detect_manipulation(phrase) == "threat"


@pytest.mark.parametrize("phrase", FLATTERY_PHRASES)
def test_flattery_phrases(phrase):
    assert md.detect_manipulation(phrase) == "excessive_flattery"


@pytest.mark.parametrize("phrase", DEFLECTION_PHRASES)
def test_deflection_phrases(phrase):
    assert md.detect_manipulation(phrase) == "deflection"


@pytest.mark.parametrize("phrase", GASLIGHTING_PHRASES)
def test_gaslighting_phrases(phrase):
    assert md.detect_manipulation(phrase) == "gaslighting"


def test_detect_manipulation_categories():
    assert md.detect_manipulation("After all I've done for you") == "guilt_tripping"
    assert md.detect_manipulation("You'll regret this") == "threat"
    assert md.detect_manipulation("You're the best!") == "excessive_flattery"
    assert md.detect_manipulation("You're overreacting") == "deflection"
    assert md.detect_manipulation("That never happened") == "gaslighting"
    assert md.detect_manipulation("Just a normal message") is None


def test_model_path_used(monkeypatch):
    class DummyClassifier:
        def __call__(self, text, truncation=True):
            return [{"label": "threat"}]

    monkeypatch.setenv("MANIP_MODEL_PATH", "dummy")
    import importlib

    tf = importlib.import_module("transformers")
    monkeypatch.setattr(
        tf, "pipeline", lambda *a, **k: DummyClassifier(), raising=False
    )

    md_reloaded = importlib.reload(md)
    try:
        assert md_reloaded.detect_manipulation("hello") == "threat"
    finally:
        monkeypatch.delenv("MANIP_MODEL_PATH", raising=False)
        importlib.reload(md)
