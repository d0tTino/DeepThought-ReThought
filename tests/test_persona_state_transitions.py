import pytest

from deepthought.services.db_manager import DBManager
from deepthought.services.persona_manager import PersonaManager


@pytest.mark.asyncio
async def test_persona_state_transitions_and_persistence_across_sessions(tmp_path):
    db = DBManager(str(tmp_path / "persona-state.db"))
    await db.init_db()
    pm = PersonaManager(db)

    start = await pm.transition_persona_state(
        "u-1",
        signals={
            "perception": {"flirtation": 0.0, "avoidance": 0.0, "manipulation": 0.0},
            "delta": 0.0,
            "affinity": 0.0,
            "trust": 0.0,
            "familiarity_tier": "low",
            "relationship_status": "neutral",
        },
    )
    assert start["current"] == "new_acquaintance"

    familiar = await pm.transition_persona_state(
        "u-1",
        signals={
            "perception": {"flirtation": 0.3, "avoidance": 0.1, "manipulation": 0.0},
            "delta": 0.5,
            "affinity": 3.0,
            "trust": 2.0,
            "familiarity_tier": "medium",
            "relationship_status": "neutral",
            "feedback": {"positive": True},
        },
    )
    assert familiar["current"] == "familiar"
    assert familiar["policy_hints"]["tone"] == "warm_consistent"

    pm2 = PersonaManager(db)
    repair = await pm2.transition_persona_state(
        "u-1",
        signals={
            "perception": {"flirtation": 0.0, "avoidance": 0.8, "manipulation": 0.5},
            "delta": -0.5,
            "affinity": 1.0,
            "trust": -2.0,
            "familiarity_tier": "medium",
            "relationship_status": "neutral",
            "feedback": {"negative": True},
        },
    )
    assert repair["current"] == "repair_mode"
    assert repair["policy_hints"]["repair_needed"] is True

    pm3 = PersonaManager(db)
    recovered = await pm3.transition_persona_state(
        "u-1",
        signals={
            "perception": {"flirtation": 0.2, "avoidance": 0.0, "manipulation": 0.0},
            "delta": 0.8,
            "affinity": 7.0,
            "trust": 8.0,
            "familiarity_tier": "high",
            "relationship_status": "friend",
            "feedback": {"positive": True},
        },
    )
    assert recovered["current"] == "trusted"
    assert recovered["policy_hints"]["tone"] == "direct_collaborative"
    assert len(recovered["evidence"]) >= 4

    await db.close()


@pytest.mark.asyncio
async def test_acceptance_tone_continuity_across_sessions(tmp_path):
    db = DBManager(str(tmp_path / "tone-continuity.db"))
    await db.init_db()

    session_one = PersonaManager(db)
    state_one = await session_one.transition_persona_state(
        "u-9",
        signals={
            "perception": {"flirtation": 0.25, "avoidance": 0.0, "manipulation": 0.0},
            "delta": 0.4,
            "affinity": 3.0,
            "trust": 3.0,
            "familiarity_tier": "medium",
        },
    )
    assert state_one["current"] == "familiar"
    assert state_one["policy_hints"]["tone"] == "warm_consistent"

    session_two = PersonaManager(db)
    state_two = await session_two.transition_persona_state(
        "u-9",
        signals={
            "perception": {"flirtation": 0.2, "avoidance": 0.0, "manipulation": 0.0},
            "delta": 0.2,
            "affinity": 3.0,
            "trust": 3.0,
            "familiarity_tier": "medium",
        },
    )
    assert state_two["current"] == "familiar"
    assert state_two["policy_hints"]["tone"] == "warm_consistent"
    assert len(state_two["evidence"]) >= 2

    await db.close()
