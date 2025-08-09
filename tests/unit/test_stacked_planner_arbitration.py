from datetime import datetime, timedelta

from deepthought.planning import StackedPlanner


class DummyTranslator:
    def translate(self, goal: str):
        return "", ""


def dummy_planner(domain: str, problem: str):
    return []


def test_should_act_balances_utilities():
    planner = StackedPlanner(DummyTranslator(), dummy_planner)
    assert planner.should_act(info_gain=1.0)
    assert not planner.should_act(info_gain=0.1, cover_risk=1.0)
    assert planner.should_act(vibes_fit=1.0)


def test_silence_threshold_and_cooldown(monkeypatch):
    import deepthought.planning.stacked_planner as sp

    fake_now = datetime(2024, 1, 1)

    class FakeDateTime(datetime):
        @classmethod
        def utcnow(cls):
            return fake_now

    monkeypatch.setattr(sp, "datetime", FakeDateTime)
    planner = sp.StackedPlanner(DummyTranslator(), dummy_planner)

    assert not planner.should_act(info_gain=0.1, silence_threshold=0.5, cooldown=10)
    assert not planner.should_act(info_gain=1.0, silence_threshold=0.0)
    fake_now = fake_now + timedelta(seconds=10)
    assert planner.should_act(info_gain=1.0, silence_threshold=0.0)


def test_should_act_silent_when_crowded_bots():
    planner = StackedPlanner(DummyTranslator(), dummy_planner)
    participants = ["AlphaBot", "BetaBot", "Charlie"]
    assert not planner.should_act(info_gain=1.0, participants=participants)
