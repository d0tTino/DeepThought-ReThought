from __future__ import annotations

from datetime import datetime

import deepthought.planning.stacked_planner as sp
from deepthought.planning.stacked_planner import StackedPlanner


class DummyTranslator:
    def translate(self, goal: str):  # pragma: no cover - dummy
        return "", ""


def dummy_planner(domain: str, problem: str):  # pragma: no cover - dummy
    return []


def test_should_act_obeys_silence_heuristic(monkeypatch, tmp_path):
    planner = StackedPlanner(DummyTranslator(), dummy_planner, snapshot_dir=tmp_path)
    planner.silence_rate = 2
    planner.silence_threshold = 0.0

    now = datetime(2024, 1, 1)

    class FakeDatetime(datetime):
        @classmethod
        def utcnow(cls):  # pragma: no cover - mocked
            return now

    monkeypatch.setattr(sp, "datetime", FakeDatetime)

    assert planner.should_act(["hi"])
    assert planner.should_act(["hi"])
    assert not planner.should_act(["hi"])


def test_should_act_checks_novelty(monkeypatch, tmp_path):
    planner = StackedPlanner(DummyTranslator(), dummy_planner, snapshot_dir=tmp_path)

    monkeypatch.setattr(sp.bot_interaction, "novel_response", lambda text, threshold: False)
    called = []
    monkeypatch.setattr(sp.bot_interaction, "record_bot_message", lambda text: called.append(text))

    assert not planner.should_act(planned_text="hello", conversation=["hi"])
    assert called == []
