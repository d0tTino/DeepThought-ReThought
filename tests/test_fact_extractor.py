import pytest

from deepthought.memory.fact_extractor import (
    extract_and_store_user_facts,
    extract_user_facts,
    format_user_facts_for_prompt,
)
from deepthought.services import DBManager


def test_extract_user_facts_patterns():
    facts = extract_user_facts(
        "You can call me Ace. My hobbies are hiking, reading, and cooking. "
        "My favorite color is blue."
    )

    assert facts == {
        "nickname": "Ace",
        "hobbies": ["hiking", "reading", "cooking"],
        "favorites": {"color": "blue"},
    }

    second_facts = extract_user_facts("Favorites are pizza, tacos, and ramen.")
    assert second_facts == {"favorites": {"general": ["pizza", "tacos", "ramen"]}}

    third_facts = extract_user_facts("In my free time I play guitar & draw.")
    assert third_facts == {"hobbies": ["play guitar", "draw"]}


@pytest.mark.asyncio
async def test_extract_and_store_user_facts_merges(tmp_path):
    pytest.importorskip("aiosqlite")
    db_path = tmp_path / "profiles.db"
    db = DBManager(str(db_path))

    await extract_and_store_user_facts(
        42,
        "Call me Ace. My hobbies are hiking and chess. My favorite color is blue.",
        db,
    )
    stored = await db.get_user_profile(42)
    assert stored == {
        "nickname": "Ace",
        "hobbies": ["hiking", "chess"],
        "favorites": {"color": "blue"},
    }

    await extract_and_store_user_facts(
        42,
        "I enjoy painting. My favorite food is sushi.",
        db,
    )
    merged = await db.get_user_profile(42)
    assert merged == {
        "nickname": "Ace",
        "hobbies": ["hiking", "chess", "painting"],
        "favorites": {"color": "blue", "food": "sushi"},
    }


def test_format_user_facts_for_prompt_summary():
    profile = {
        "nickname": "Ace",
        "hobbies": ["hiking", "reading"],
        "favorites": {"color": "blue", "food": "sushi"},
    }

    summary = format_user_facts_for_prompt(profile)

    assert (
        summary
        == "User facts - Nickname: Ace | Hobbies: hiking, reading | Favorites: color: blue, food: sushi"
    )
