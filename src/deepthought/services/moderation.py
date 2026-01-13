import logging
import os
import re
from importlib import import_module
from typing import Callable, Iterable, Mapping, MutableMapping, Sequence, Tuple

logger = logging.getLogger(__name__)

# Simple list of banned phrases that should not be processed further
# These include generic banned words plus phrases related to credential or
# secret requests.
CREDENTIAL_PHRASES = [
    "password",
    "passcode",
    "login",
    "username",
    "user name",
    "credentials",
    "credit card",
    "bank account",
    "ssn",
    "social security",
    "api key",
    "token",
    "secret key",
    "private key",
]

SECRET_REQUEST_PHRASES = [
    "secret",
    "confidential",
    "private information",
]

BANNED_PHRASES = [
    "banned",
    "prohibited",
    *CREDENTIAL_PHRASES,
    *SECRET_REQUEST_PHRASES,
]

# Configuration toggles for moderation behavior.
MODERATION_LOG_DECISIONS = os.getenv("MODERATION_LOG_DECISIONS", "true").lower() in {
    "true",
    "1",
    "yes",
}
TOXICITY_CHECK_ENABLED = os.getenv("MODERATION_TOXICITY_ENABLED", "true").lower() in {
    "true",
    "1",
    "yes",
}
TOXICITY_THRESHOLD = float(os.getenv("MODERATION_TOXICITY_THRESHOLD", "0.6"))
TOXICITY_CLASSIFIER_PATH = os.getenv("MODERATION_TOXICITY_CLASSIFIER", "")

# Lightweight toxicity lexicon with severity weights. Users can add terms via
# ``MODERATION_PROFANITY_LIST``, using comma-separated entries. Each entry can
# optionally include a weight (``term:weight``). Example::
#
#     export MODERATION_PROFANITY_LIST="foobar:0.9,baz"
DEFAULT_TOXICITY_TERMS: Mapping[str, float] = {
    "idiot": 0.4,
    "stupid": 0.4,
    "loser": 0.4,
    "dumb": 0.4,
    "ugly": 0.4,
    "moron": 0.5,
    "jerk": 0.5,
    "shut up": 0.6,
    "trash": 0.5,
    "worthless": 0.7,
    "hate you": 0.7,
    "piece of": 0.6,
    "kill yourself": 1.0,
}

ToxicityClassifier = Callable[[str], object]


def _load_toxicity_classifier(path: str) -> ToxicityClassifier:
    module_path, _, attr_name = path.rpartition(":")
    if not module_path or not attr_name:
        raise ValueError("Toxicity classifier path must be in 'module:attribute' format")
    module = import_module(module_path)
    classifier = getattr(module, attr_name)
    if not callable(classifier):
        raise TypeError(f"Toxicity classifier {path!r} is not callable")
    return classifier


def _parse_classifier_result(result: object) -> Tuple[float | None, Sequence[str]]:
    matches: list[str] = []
    score: float | None = None
    score_value: object | None = None
    matches_value: object | None = None
    if isinstance(result, tuple) and len(result) == 2:
        score_value, matches_value = result
    elif isinstance(result, Mapping):
        score_value = result.get("score") or result.get("toxicity") or result.get("value")
        matches_value = result.get("matches") or result.get("terms") or result.get("labels")
    else:
        score_value = result
    try:
        if score_value is not None:
            score = float(score_value)
    except (TypeError, ValueError):
        score = None
    if isinstance(matches_value, str):
        matches = [matches_value]
    elif isinstance(matches_value, Sequence) and not isinstance(matches_value, (str, bytes)):
        matches = [str(item) for item in matches_value if item]
    if score is not None:
        score = max(0.0, min(score, 1.0))
    return score, matches


def _parse_weighted_terms(raw_terms: str) -> MutableMapping[str, float]:
    parsed: MutableMapping[str, float] = {}
    for chunk in raw_terms.split(","):
        if not chunk:
            continue
        if ":" in chunk:
            phrase, _, weight_str = chunk.partition(":")
            try:
                weight = float(weight_str)
            except ValueError:
                weight = 0.6
        else:
            phrase, weight = chunk, 0.6
        normalized = phrase.strip().lower()
        if not normalized:
            continue
        parsed[normalized] = max(0.1, min(weight, 1.0))
    return parsed


def _build_toxicity_terms() -> MutableMapping[str, float]:
    configured = os.getenv("MODERATION_PROFANITY_LIST", "")
    combined: MutableMapping[str, float] = dict(DEFAULT_TOXICITY_TERMS)
    if configured:
        combined.update(_parse_weighted_terms(configured))
    return combined


TOXICITY_TERMS: Mapping[str, float] = _build_toxicity_terms()
_MAX_TOXICITY_WEIGHT = max(TOXICITY_TERMS.values(), default=1.0)
_TOXICITY_CLASSIFIER: ToxicityClassifier | None = None
if TOXICITY_CLASSIFIER_PATH:
    try:
        _TOXICITY_CLASSIFIER = _load_toxicity_classifier(TOXICITY_CLASSIFIER_PATH)
    except Exception:
        logger.warning(
            "Unable to load toxicity classifier %s; falling back to lexicon scoring",
            TOXICITY_CLASSIFIER_PATH,
            exc_info=True,
        )


def evaluate_toxicity(
    text: str,
    *,
    terms: Mapping[str, float] | None = None,
) -> Tuple[float, Sequence[str]]:
    """Return a toxicity score and matched terms for ``text``."""
    if not isinstance(text, str):
        return 0.0, []
    if _TOXICITY_CLASSIFIER is not None:
        try:
            score, matches = _parse_classifier_result(_TOXICITY_CLASSIFIER(text))
            if score is not None:
                return score, matches
        except Exception:
            logger.warning(
                "Toxicity classifier failed; falling back to lexicon scoring",
                exc_info=True,
            )
    lexicon = terms or TOXICITY_TERMS
    lowered = text.lower()
    matches: list[str] = []
    total_weight = 0.0
    for phrase, weight in lexicon.items():
        if " " in phrase:
            found = phrase in lowered
        else:
            found = re.search(rf"\b{re.escape(phrase)}\b", lowered) is not None
        if found:
            matches.append(phrase)
            total_weight += weight
    if not matches:
        return 0.0, []
    score = min(1.0, total_weight / (_MAX_TOXICITY_WEIGHT * len(matches)))
    return score, matches


def get_toxicity_terms() -> Mapping[str, float]:
    """Return the configured toxicity lexicon."""

    return TOXICITY_TERMS


def get_profanity_list() -> Tuple[str, ...]:
    """Return profanity phrases for outbound filtering or UI hints."""

    return tuple(TOXICITY_TERMS.keys())


def _log_decision(
    *,
    text: str,
    allowed: bool,
    banned_matches: Sequence[str],
    toxicity_score: float | None = None,
    toxicity_matches: Sequence[str] | None = None,
    toxicity_threshold: float | None = None,
) -> None:
    if not MODERATION_LOG_DECISIONS:
        return
    logger.info(
        "Moderation decision: allowed=%s banned_matches=%s toxicity_score=%s "
        "toxicity_matches=%s toxicity_threshold=%s text_preview=%s",
        allowed,
        list(banned_matches),
        None if toxicity_score is None else round(toxicity_score, 3),
        list(toxicity_matches or []),
        toxicity_threshold,
        text[:160].replace("\n", " "),
    )


def is_allowed(
    text: str,
    banned_phrases: Iterable[str] | None = None,
    *,
    check_toxicity: bool | None = None,
    toxicity_threshold: float | None = None,
) -> bool:
    """Return ``True`` if ``text`` does not contain any banned phrases."""
    if not isinstance(text, str):
        return False
    phrases = list(banned_phrases) if banned_phrases is not None else BANNED_PHRASES
    lowered = text.lower()
    banned_matches = [phrase for phrase in phrases if phrase in lowered]
    if banned_matches:
        _log_decision(
            text=text,
            allowed=False,
            banned_matches=banned_matches,
        )
        return False
    use_toxicity = TOXICITY_CHECK_ENABLED if check_toxicity is None else check_toxicity
    threshold = TOXICITY_THRESHOLD if toxicity_threshold is None else toxicity_threshold
    toxicity_score = None
    toxicity_matches: Sequence[str] = []
    if use_toxicity:
        toxicity_score, toxicity_matches = evaluate_toxicity(text)
        allowed = toxicity_score < threshold
    else:
        allowed = True
    _log_decision(
        text=text,
        allowed=allowed,
        banned_matches=banned_matches,
        toxicity_score=toxicity_score,
        toxicity_matches=toxicity_matches,
        toxicity_threshold=threshold if use_toxicity else None,
    )
    return allowed
