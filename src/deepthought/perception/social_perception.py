from __future__ import annotations

"""Simple transformer based social cue classifier."""

import logging
import os
from typing import Dict

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_PATH = os.getenv("SOCIAL_PERCEPTION_MODEL", "path/to/social-perception-model")
LABELS = ["flirtation", "avoidance", "manipulation"]

_tokenizer: AutoTokenizer | None = None
_model: AutoModelForSequenceClassification | None = None


logger = logging.getLogger(__name__)


def _load() -> bool:
    """Load model and tokenizer if available.

    Returns ``True`` if the model is ready for use, otherwise ``False``.
    """
    global _tokenizer, _model

    if _tokenizer is not None and _model is not None:
        return True

    if not MODEL_PATH or MODEL_PATH == "path/to/social-perception-model":
        logger.warning(
            "SOCIAL_PERCEPTION_MODEL not set. Returning neutral perception scores."
        )
        return False

    if not os.path.exists(MODEL_PATH):
        logger.warning(
            "Social perception model path not found: %s. Returning neutral perception scores.",
            MODEL_PATH,
        )
        return False

    try:
        if _tokenizer is None:
            _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        if _model is None:
            _model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        return True
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Failed to load social perception model: %s", e, exc_info=True)
        _tokenizer = None
        _model = None
        return False


def analyze(text: str) -> Dict[str, float]:
    """Return probabilities for flirtation, avoidance and manipulation."""
    if not _load():
        neutral = 1.0 / len(LABELS)
        return {label: neutral for label in LABELS}

    assert _tokenizer and _model
    inputs = _tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        logits = _model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0].tolist()
    return {label: float(prob) for label, prob in zip(LABELS, probs)}
