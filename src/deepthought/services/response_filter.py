"""Lightweight response filtering utilities."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from importlib import import_module
from typing import Callable, Iterable, Mapping

from deepthought.services import moderation

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
    replacement: str = "***"
    toxicity_threshold: float | None = moderation.TOXICITY_THRESHOLD
    toxicity_terms: Mapping[str, float] | None = None

    def _log_decision(
        self,
        *,
        allowed: bool,
        reason: str,
        text: str,
        details: Mapping[str, object] | None = None,
    ) -> None:
        if not moderation.MODERATION_LOG_DECISIONS:
            return
        logger.info(
            "Response filter decision: allowed=%s reason=%s details=%s text_preview=%s",
            allowed,
            reason,
            {} if details is None else details,
            text[:160].replace("\n", " "),
        )

    def is_safe(self, text: str) -> bool:
        sanitized = self.sanitize(text)
        return sanitized is not None

    def sanitize(self, text: str) -> str | None:
        """Return sanitized ``text`` or ``None`` if it should be blocked."""

        if not isinstance(text, str):
            return None

        sanitized = text
        if self.denylist:
            pattern = re.compile(
                "|".join(re.escape(term) for term in self.denylist), re.IGNORECASE
            )
            matches = pattern.findall(sanitized)

            def _sub(match: re.Match[str]) -> str:
                return self.replacement

            sanitized = pattern.sub(_sub, sanitized)
            if matches:
                self._log_decision(
                    allowed=True,
                    reason="denylist_replace",
                    text=sanitized,
                    details={"matches": sorted({match.lower() for match in matches})},
                )

        if self.classifier is not None:
            try:
                if not self.classifier(sanitized):
                    self._log_decision(
                        allowed=False,
                        reason="classifier_blocked",
                        text=sanitized,
                    )
                    return None
            except Exception:
                logger.warning(
                    "Response classifier failed; treating as unsafe", exc_info=True
                )
                self._log_decision(
                    allowed=False,
                    reason="classifier_error",
                    text=sanitized,
                )
                return None

        if self.toxicity_threshold is not None:
            score, matches = moderation.evaluate_toxicity(
                sanitized, terms=self.toxicity_terms
            )
            if score >= self.toxicity_threshold:
                logger.info(
                    "Blocked response due to toxicity score %.3f and matches %s",
                    score,
                    matches,
                )
                self._log_decision(
                    allowed=False,
                    reason="toxicity_blocked",
                    text=sanitized,
                    details={
                        "score": round(score, 3),
                        "threshold": self.toxicity_threshold,
                        "matches": list(matches),
                    },
                )
                return None
            self._log_decision(
                allowed=True,
                reason="toxicity_checked",
                text=sanitized,
                details={
                    "score": round(score, 3),
                    "threshold": self.toxicity_threshold,
                    "matches": list(matches),
                },
            )

        return sanitized


def build_response_filter(
    denylist: Iterable[str] | None = None,
    classifier_path: str | None = None,
    *,
    replacement: str = "***",
    toxicity_threshold: float | None = None,
    toxicity_terms: Mapping[str, float] | None = None,
) -> ResponseFilter:
    """Construct a response filter for the configured inputs."""

    env_denylist = os.getenv("RESPONSE_FILTER_DENYLIST", "")
    configured_denylist = _normalize_denylist(
        env_denylist.split(",") if env_denylist else []
    )
    normalized = _normalize_denylist(denylist or ()) + configured_denylist
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
    return ResponseFilter(
        denylist=normalized,
        classifier=classifier,
        replacement=replacement,
        toxicity_threshold=toxicity_threshold
        if toxicity_threshold is not None
        else moderation.TOXICITY_THRESHOLD,
        toxicity_terms=toxicity_terms or moderation.get_toxicity_terms(),
    )
