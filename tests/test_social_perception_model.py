import importlib
import deepthought.config as config
import deepthought.perception.social_perception as sp


def test_default_model_detects_cues(monkeypatch):
    monkeypatch.delenv("SOCIAL_PERCEPTION_MODEL", raising=False)
    config._settings_cache = None
    importlib.reload(sp)

    cases = [
        ("I love you", "flirtation"),
        ("Please leave me alone", "avoidance"),
        ("You must obey me", "manipulation"),
        ("yeah right, that will work", "sarcasm"),
        ("Thanks for your help", "supportiveness"),
    ]

    for text, label in cases:
        scores = sp.analyze(text)
        assert max(scores, key=scores.get) == label
