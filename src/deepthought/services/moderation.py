import logging
import os
import re
from typing import Iterable, Mapping, Sequence, Tuple

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

# Lightweight toxicity lexicon with severity weights.
TOXICITY_TERMS: Mapping[str, float] = {
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
_MAX_TOXICITY_WEIGHT = max(TOXICITY_TERMS.values(), default=1.0)


def evaluate_toxicity(
    text: str,
    *,
    terms: Mapping[str, float] | None = None,
) -> Tuple[float, Sequence[str]]:
    """Return a toxicity score and matched terms for ``text``."""
    if not isinstance(text, str):
        return 0.0, []
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
