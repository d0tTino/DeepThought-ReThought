import json
from pathlib import Path

from deepthought.planning.stacked_planner import StackedPlanner


class DummyTranslator:
    def translate(self, goal: str):
        return "domain", "problem"


def dummy_planner(domain: str, problem: str):
    return ["move", "scan", "report"]


def test_layers_modify_actions_and_persist(tmp_path: Path):
    planner = StackedPlanner(
        translator=DummyTranslator(),
        planner_fn=dummy_planner,
        snapshot_dir=tmp_path,
    )

    plan = planner.generate_plan("test")
    assert plan == ["REACT:MOVE?", "REACT:SCAN?", "REACT:REPORT?"]

    reactive_file = next(tmp_path.glob("*_reactive.json"))
    reactive = json.loads(reactive_file.read_text())
    assert reactive["actions"] == ["react:move", "react:scan", "react:report"]

    investigative_file = next(tmp_path.glob("*_investigative.json"))
    investigative = json.loads(investigative_file.read_text())
    assert investigative["actions"] == [
        "react:move?",
        "react:scan?",
        "react:report?",
    ]

    arc_file = next(tmp_path.glob("*_arc.json"))
    arc = json.loads(arc_file.read_text())
    assert arc["actions"] == plan


def test_high_low_utility_decision():
    planner = StackedPlanner(translator=DummyTranslator(), planner_fn=dummy_planner)
    planner.silence_threshold = 0.5

    low_ctx = ["hello world"]
    high_ctx = ["please vibe cool awesome thanks"]

    assert not planner.should_act(low_ctx)
    assert planner.should_act(high_ctx)


def test_silence_heuristic_blocks_spam():
    planner = StackedPlanner(translator=DummyTranslator(), planner_fn=dummy_planner)
    planner.silence_rate = 2
    conv = ["hello world"]

    assert planner.should_act(conv)
    assert planner.should_act(conv)
    assert not planner.should_act(conv)
