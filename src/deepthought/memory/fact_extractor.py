"""Extract and persist user facts for prompt building."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

from ..services.db_manager import DBManager

NICKNAME_PATTERNS = [
    re.compile(
        r"\b(?:call me|i go by|my nickname is|my nick(?:name)? is|you can call me)\s+(?P<name>[^.!?,]+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:i am|i'm)\s+known as\s+(?P<name>[^.!?,]+)",
        re.IGNORECASE,
    ),
]

HOBBY_PATTERNS = [
    re.compile(
        r"\bmy hobbies are\s+(?P<hobbies>[^.!?]+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bmy hobby is\s+(?P<hobbies>[^.!?]+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bi (?:enjoy|love|like)\s+(?P<hobbies>[^.!?]+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bin my free time\s+i\s+(?P<hobbies>[^.!?]+)",
        re.IGNORECASE,
    ),
]

FAVORITE_PATTERNS = [
    re.compile(
        r"\b(?:my\s+)?favorite\s+(?P<category>[a-zA-Z ]+?)\s+(?:is|are|:)\s*(?P<value>[^.!?]+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:my\s+)?favorites\s+(?:are|:)\s*(?P<value>[^.!?]+)",
        re.IGNORECASE,
    ),
]

TEMPORAL_EVENT_PATTERNS = [
    re.compile(r"\b(?:tomorrow|next week|next month|on \w+)\b[^.!?]*", re.IGNORECASE),
    re.compile(r"\bi\s+(?:will|am going to)\s+([^.!?]+)", re.IGNORECASE),
]


def _split_list(text: str) -> list[str]:
    parts = re.split(r"\s*(?:,|and|&|/)\s*", text.strip())
    return [part.strip() for part in parts if part.strip()]


def _extract_nickname(message: str) -> str | None:
    for pattern in NICKNAME_PATTERNS:
        match = pattern.search(message)
        if match:
            name = match.group("name").strip(" \"'")
            return name or None
    return None


def _extract_hobbies(message: str) -> list[str]:
    hobbies: list[str] = []
    for pattern in HOBBY_PATTERNS:
        match = pattern.search(message)
        if match:
            hobbies.extend(_split_list(match.group("hobbies")))
    return hobbies


def _extract_favorites(message: str) -> dict[str, list[str] | str]:
    favorites: dict[str, list[str] | str] = {}
    for pattern in FAVORITE_PATTERNS:
        match = pattern.search(message)
        if not match:
            continue
        if "category" in match.groupdict() and match.group("category"):
            category = match.group("category").strip().lower()
            value = match.group("value").strip()
            favorites[category] = value
        else:
            favorites["general"] = _split_list(match.group("value"))
    return favorites


def extract_user_facts(message: str) -> dict[str, Any]:
    """Extract nickname, hobbies, and favorites from a single message."""
    nickname = _extract_nickname(message)
    hobbies = _extract_hobbies(message)
    favorites = _extract_favorites(message)
    facts: dict[str, Any] = {}
    if nickname:
        facts["nickname"] = nickname
    if hobbies:
        facts["hobbies"] = hobbies
    if favorites:
        facts["favorites"] = favorites
    return facts


def extract_typed_fact_triples_from_turn(
    *,
    user_id: str,
    message: str,
    timestamp: str,
    source_id: str,
) -> list[dict[str, Any]]:
    """Extract typed triples+attributes from a single conversation turn."""

    facts = extract_user_facts(message)
    triples: list[dict[str, Any]] = []
    if "nickname" in facts:
        triples.append(
            {
                "subject_id": str(user_id),
                "subject_type": "user",
                "predicate": "has_nickname",
                "object_id": f"nickname:{facts['nickname'].lower()}",
                "object_type": "nickname",
                "object_value": str(facts["nickname"]),
                "attributes": {"source_id": source_id, "timestamp": timestamp},
                "confidence": 0.92,
                "fact_type": "profile",
            }
        )
    for hobby in facts.get("hobbies", []):
        triples.append(
            {
                "subject_id": str(user_id),
                "subject_type": "user",
                "predicate": "likes_hobby",
                "object_id": f"hobby:{str(hobby).lower()}",
                "object_type": "hobby",
                "object_value": str(hobby),
                "attributes": {"source_id": source_id, "timestamp": timestamp},
                "confidence": 0.8,
                "fact_type": "preference",
            }
        )
    favorites = facts.get("favorites", {})
    if isinstance(favorites, Mapping):
        for category, value in favorites.items():
            values = value if isinstance(value, list) else [value]
            for item in values:
                triples.append(
                    {
                        "subject_id": str(user_id),
                        "subject_type": "user",
                        "predicate": "favorite",
                        "object_id": f"favorite:{category}:{str(item).lower()}",
                        "object_type": "favorite",
                        "object_value": str(item),
                        "attributes": {
                            "category": category,
                            "source_id": source_id,
                            "timestamp": timestamp,
                        },
                        "confidence": 0.83,
                        "fact_type": "preference",
                    }
                )
    if message.strip():
        triples.append(
            {
                "subject_id": str(user_id),
                "subject_type": "user",
                "predicate": "mentioned",
                "object_id": None,
                "object_type": "utterance",
                "object_value": message.strip(),
                "attributes": {"source_id": source_id, "timestamp": timestamp},
                "confidence": 0.55,
                "fact_type": "utterance",
            }
        )
    for pattern in TEMPORAL_EVENT_PATTERNS:
        match = pattern.search(message)
        if not match:
            continue
        event_text = match.group(0).strip()
        triples.append(
            {
                "subject_id": str(user_id),
                "subject_type": "user",
                "predicate": "plans_event",
                "object_id": None,
                "object_type": "event",
                "object_value": event_text,
                "attributes": {
                    "source_id": source_id,
                    "timestamp": timestamp,
                    "temporal": True,
                },
                "confidence": 0.76,
                "fact_type": "temporal_fact",
            }
        )
        break
    return triples


def _merge_list(existing: Iterable[str] | None, incoming: Iterable[str]) -> list[str]:
    merged = list(dict.fromkeys([*(existing or []), *incoming]))
    return [item for item in merged if item]


def merge_user_profiles(
    existing: Mapping[str, Any] | list | str | None,
    updates: Mapping[str, Any],
) -> dict[str, Any]:
    profile: dict[str, Any] = {}
    if isinstance(existing, Mapping):
        profile.update(existing)
    elif existing is not None:
        profile["raw"] = existing

    if "nickname" in updates:
        profile["nickname"] = updates["nickname"]

    if "hobbies" in updates:
        profile["hobbies"] = _merge_list(profile.get("hobbies"), updates["hobbies"])

    if "favorites" in updates:
        favorites = profile.get("favorites")
        merged_favorites: dict[str, Any] = {}
        if isinstance(favorites, Mapping):
            merged_favorites.update(favorites)
        for key, value in updates["favorites"].items():
            merged_favorites[key] = value
        profile["favorites"] = merged_favorites

    return profile


async def extract_and_store_user_facts(
    user_id: int,
    message: str,
    db_manager: DBManager | None = None,
) -> dict[str, Any]:
    """Extract facts from a message and persist them to ``user_profiles``."""
    facts = extract_user_facts(message)
    if not facts:
        return {}
    db = db_manager or DBManager()
    existing = await db.get_user_profile(user_id)
    merged = merge_user_profiles(existing, facts)
    await db.set_user_profile(user_id, merged)
    return facts


async def get_user_fact_profile(
    user_id: int,
    db_manager: DBManager | None = None,
) -> dict[str, Any] | None:
    """Return the stored user fact profile as a dict when possible."""
    db = db_manager or DBManager()
    profile = await db.get_user_profile(user_id)
    if profile is None:
        return None
    if isinstance(profile, Mapping):
        return dict(profile)
    return {"raw": profile}


def format_user_facts_for_prompt(profile: Mapping[str, Any] | None) -> str | None:
    """Format a stored user profile into a prompt-ready summary."""
    if not profile:
        return None
    parts: list[str] = []
    nickname = profile.get("nickname")
    if nickname:
        parts.append(f"Nickname: {nickname}")
    hobbies = profile.get("hobbies")
    if hobbies:
        if isinstance(hobbies, str):
            parts.append(f"Hobbies: {hobbies}")
        else:
            parts.append(f"Hobbies: {', '.join(hobbies)}")
    favorites = profile.get("favorites")
    if isinstance(favorites, Mapping) and favorites:
        fav_parts = [f"{key}: {value}" for key, value in favorites.items()]
        parts.append(f"Favorites: {', '.join(fav_parts)}")
    if not parts:
        return None
    return "User facts - " + " | ".join(parts)


async def build_user_fact_context(
    user_id: int,
    db_manager: DBManager | None = None,
) -> str | None:
    """Return a formatted user-fact string for prompt building."""
    profile = await get_user_fact_profile(user_id, db_manager=db_manager)
    return format_user_facts_for_prompt(profile)
