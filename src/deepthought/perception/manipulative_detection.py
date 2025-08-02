from __future__ import annotations

"""Detect manipulative language using heuristics or an optional model.

Set the :envvar:`MANIP_MODEL_PATH` environment variable to load a
Transformer-based text classifier. If unset, simple phrase heuristics
are applied instead.
"""

import os
from typing import Optional

_classifier = None
_model_checked = False

# Simple lists of phrases for each manipulation tactic
GUILT_TRIP_PHRASES = [
    "after all i've done for you",
    "you owe me",
    "i thought you cared",
    "if you really loved me",
    "how could you do this",
]

THREAT_PHRASES = [
    "or else",
    "you'll regret",
    "i'll make you",
    "i will hurt",
    "i will harm",
    "i'm going to report",
]

FLATTERY_PHRASES = [
    "you're the best",
    "no one is as",
    "you're amazing",
    "you're incredible",
    "you're perfect",
    "trust me",
]

DEFLECTION_PHRASES = [
    "you're overreacting",
    "you're taking this too seriously",
    "let's not dwell on",
    "that's not important",
]

GASLIGHTING_PHRASES = [
    "that never happened",
    "you're imagining things",
    "you're making things up",
    "i never said that",
]


def _get_classifier():
    """Return a text classification pipeline if available."""
    global _classifier, _model_checked
    if _model_checked:
        return _classifier
    _model_checked = True
    model_path = os.getenv("MANIP_MODEL_PATH")
    if not model_path:
        return None
    try:  # pragma: no cover - optional dependency
        from transformers import pipeline
    except Exception:
        return None
    try:  # pragma: no cover - optional dependency
        _classifier = pipeline("text-classification", model=model_path)
    except Exception:
        _classifier = None
    return _classifier


def detect_manipulation(text: str) -> Optional[str]:
    """Return the manipulation category using a model or heuristics."""
    classifier = _get_classifier()
    if classifier is not None:
        try:
            result = classifier(text, truncation=True)
            if isinstance(result, list) and result:
                label = str(result[0].get("label", "")).lower()
                if label and label != "none":
                    return label
        except Exception:  # pragma: no cover - model inference failure
            pass

    lower = text.lower()
    for phrase in GUILT_TRIP_PHRASES:
        if phrase in lower:
            return "guilt_tripping"
    for phrase in THREAT_PHRASES:
        if phrase in lower:
            return "threat"
    for phrase in FLATTERY_PHRASES:
        if phrase in lower:
            return "excessive_flattery"
    for phrase in DEFLECTION_PHRASES:
        if phrase in lower:
            return "deflection"
    for phrase in GASLIGHTING_PHRASES:
        if phrase in lower:
            return "gaslighting"
    return None
