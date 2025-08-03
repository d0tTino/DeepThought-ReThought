from __future__ import annotations

"""Simple transformer based social cue classifier."""

import json
import logging
import os
from typing import Dict

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..config import get_settings

LABELS = ["flirtation", "avoidance", "manipulation", "sarcasm", "supportiveness"]

_tokenizer: AutoTokenizer | None = None
_model: AutoModelForSequenceClassification | None = None
_keyword_model: Dict[str, list[str]] | None = None


logger = logging.getLogger(__name__)


def _load() -> bool:
    """Load model and tokenizer if available.

    Returns ``True`` if the model is ready for use, otherwise ``False``.
    """
    global _tokenizer, _model, _keyword_model

    if (_tokenizer is not None and _model is not None) or _keyword_model is not None:
        return True

    model_path = get_settings().social_perception_model
    if model_path and os.path.exists(model_path):
        if model_path.endswith(".json"):
            try:
                with open(model_path, "r", encoding="utf-8") as f:
                    _keyword_model = json.load(f)
                return True
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(
                    "Failed to load default social perception rules: %s", e, exc_info=True
                )
                return False
        try:
            _tokenizer = AutoTokenizer.from_pretrained(model_path)
            _model = AutoModelForSequenceClassification.from_pretrained(model_path)
            return True
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("Failed to load social perception model: %s", e, exc_info=True)
            _tokenizer = None
            _model = None
            return False

    if model_path:
        logger.warning(
            "Social perception model path not found: %s. Returning neutral perception scores.",
            model_path,
        )
    else:
        logger.warning("SOCIAL_PERCEPTION_MODEL not set. Returning neutral perception scores.")
    return False


def analyze(text: str) -> Dict[str, float]:
    """Return probabilities for supported social cues."""
    if not _load():
        neutral = 1.0 / len(LABELS)
        return {label: neutral for label in LABELS}

    if _keyword_model is not None:
        lower = text.lower()
        hits = [label for label, words in _keyword_model.items() if any(w in lower for w in words)]
        if not hits:
            neutral = 1.0 / len(LABELS)
            return {label: neutral for label in LABELS}
        prob = 1.0 / len(hits)
        return {label: (prob if label in hits else 0.0) for label in LABELS}

    if _tokenizer is None or _model is None:
        neutral = 1.0 / len(LABELS)
        return {label: neutral for label in LABELS}

    inputs = _tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        logits = _model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0].tolist()
    return {label: float(prob) for label, prob in zip(LABELS, probs)}
