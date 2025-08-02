from __future__ import annotations

"""Simple transformer based social cue classifier."""

import logging
import os
from typing import Dict

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..config import get_settings

LABELS = ["flirtation", "avoidance", "manipulation", "sarcasm", "supportiveness"]

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

    model_path = get_settings().social_perception_model
    if not model_path:
        logger.warning("SOCIAL_PERCEPTION_MODEL not set. Returning neutral perception scores.")
        return False

    if not os.path.exists(model_path):
        logger.warning(
            "Social perception model path not found: %s. Returning neutral perception scores.",
            model_path,
        )
        return False

    try:
        if _tokenizer is None:
            _tokenizer = AutoTokenizer.from_pretrained(model_path)
        if _model is None:
            _model = AutoModelForSequenceClassification.from_pretrained(model_path)
        return True
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Failed to load social perception model: %s", e, exc_info=True)
        _tokenizer = None
        _model = None
        return False


def analyze(text: str) -> Dict[str, float]:
    """Return probabilities for supported social cues."""
    if not _load() or _tokenizer is None or _model is None:
        neutral = 1.0 / len(LABELS)
        return {label: neutral for label in LABELS}

    inputs = _tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        logits = _model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0].tolist()
    return {label: float(prob) for label, prob in zip(LABELS, probs)}
