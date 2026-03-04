from deepthought.services.human_interaction_policy import HumanInteractionPolicy


def test_policy_delay_bounds_are_clamped():
    policy = HumanInteractionPolicy(min_delay_seconds=0.2, max_delay_seconds=1.1)
    low = policy.decide(message_text="hi", channel_pace=0.0, familiarity=1.0)
    high = policy.decide(message_text="x" * 1000, channel_pace=999.0, familiarity=0.0)

    assert 0.2 <= low.delay_seconds <= 1.1
    assert 0.2 <= high.delay_seconds <= 1.1


def test_policy_metadata_override_is_deterministic():
    policy = HumanInteractionPolicy(max_typing_seconds=3.0)
    decision = policy.decide(
        message_text="hello",
        channel_pace=5.0,
        familiarity=0.3,
        metadata={
            "delay_seconds": 0.5,
            "typing_seconds": 5.0,
            "cooldown_seconds": 2.5,
            "style_modifiers": ["gentle", "concise"],
        },
    )

    assert decision.delay_seconds == 0.5
    assert decision.typing_seconds == 3.0
    assert decision.cooldown_seconds == 2.5
    assert decision.style_modifiers == ["gentle", "concise"]
