"""Lightweight response filtering utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib import import_module
from typing import Callable, Iterable

logger = logging.getLogger(__name__)

Classifier = Callable[[str], bool]


def _normalize_denylist(denylist: Iterable[str]) -> tuple[str, ...]:
    cleaned: list[str] = []
    for phrase in denylist:
        if not isinstance(phrase, str):
            continue
        normalized = phrase.strip().lower()
        if normalized:
            cleaned.append(normalized)
    return tuple(cleaned)


def load_classifier(classifier_path: str) -> Classifier:
    """Load a classifier callable from ``module:attribute`` notation."""
    module_path, _, attr_name = classifier_path.rpartition(":")
    if not module_path or not attr_name:
        raise ValueError("Classifier path must be in 'module:attribute' format")
    module = import_module(module_path)
    classifier = getattr(module, attr_name)
    if not callable(classifier):
        raise TypeError(f"Classifier {classifier_path!r} is not callable")
    return classifier


@dataclass(frozen=True)
class ResponseFilter:
    """Check responses against a denylist and optional classifier."""

    denylist: tuple[str, ...] = ()
    classifier: Classifier | None = None

    def is_safe(self, text: str) -> bool:
        if not isinstance(text, str):
            return False
        lowered = text.lower()
        if self.denylist and any(phrase in lowered for phrase in self.denylist):
            return False
        if self.classifier is None:
            return True
        try:
            return bool(self.classifier(text))
        except Exception:
            logger.warning("Response classifier failed; treating as unsafe", exc_info=True)
            return False


def build_response_filter(
    denylist: Iterable[str] | None = None,
    classifier_path: str | None = None,
) -> ResponseFilter:
    """Construct a response filter for the configured inputs."""
    normalized = _normalize_denylist(denylist or [])
    classifier: Classifier | None = None
    if classifier_path:
        try:
            classifier = load_classifier(classifier_path)
        except Exception:
            logger.warning(
                "Unable to load response classifier %s",
                classifier_path,
                exc_info=True,
            )
    return ResponseFilter(denylist=normalized, classifier=classifier)
