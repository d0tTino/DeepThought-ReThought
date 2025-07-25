from __future__ import annotations

"""Simple transformer based social cue classifier."""

from typing import Dict

import os

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_PATH = os.getenv("SOCIAL_PERCEPTION_MODEL", "path/to/social-perception-model")
LABELS = ["flirtation", "avoidance", "manipulation"]

_tokenizer: AutoTokenizer | None = None
_model: AutoModelForSequenceClassification | None = None


def _load() -> None:
    """Load model and tokenizer if available."""
    global _tokenizer, _model

    if _tokenizer is not None and _model is not None:
        return

    if not MODEL_PATH or MODEL_PATH == "path/to/social-perception-model":
        raise RuntimeError(
            "SOCIAL_PERCEPTION_MODEL environment variable not set or empty"
        )

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model path not found: {MODEL_PATH}")

    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if _model is None:
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)


def analyze(text: str) -> Dict[str, float]:
    """Return probabilities for flirtation, avoidance and manipulation."""
    _load()
    assert _tokenizer and _model
    inputs = _tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        logits = _model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0].tolist()
    return {label: float(prob) for label, prob in zip(LABELS, probs)}
