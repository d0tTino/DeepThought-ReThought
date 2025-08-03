from __future__ import annotations

"""Detect manipulative language using heuristics or an optional model.

Set the :envvar:`MANIP_MODEL_PATH` environment variable to load a
Transformer-based text classifier. If unset, simple phrase heuristics
are applied instead.
"""

import os
from typing import Optional

from .manipulative_phrases import (  # noqa: F401
    CATEGORY_PHRASES,
    DEFLECTION_PHRASES,
    FLATTERY_PHRASES,
    GASLIGHTING_PHRASES,
    GUILT_TRIP_PHRASES,
    THREAT_PHRASES,
)

_classifier = None
_model_checked = False


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
    for category, phrases in CATEGORY_PHRASES.items():
        for phrase in phrases:
            if phrase in lower:
                return category
    return None
