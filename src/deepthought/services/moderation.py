import logging
from typing import Iterable

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


def is_allowed(text: str, banned_phrases: Iterable[str] | None = None) -> bool:
    """Return ``True`` if ``text`` does not contain any banned phrases."""
    if not isinstance(text, str):
        return False
    phrases = list(banned_phrases) if banned_phrases is not None else BANNED_PHRASES
    lowered = text.lower()
    return not any(phrase in lowered for phrase in phrases)
