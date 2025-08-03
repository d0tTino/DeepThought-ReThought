import pytest

from deepthought.affinity import AffinityTracker
from deepthought.goal_scheduler import GoalScheduler
from deepthought.persona import PersonaSelector


def test_persona_selection():
    selector = PersonaSelector({"friendly": ["hello", "hi"], "formal": ["dear", "regards"]})
    assert selector.choose("hello there") == "friendly"
    assert selector.choose("Dear sir, regards") == "formal"
    assert selector.choose("no keywords here") is None


def test_affinity_updates():
    tracker = AffinityTracker()
    tracker.update("friendly")
    tracker.update("friendly", 2)
    tracker.update("formal")
    assert tracker.get("friendly") == 3
    assert tracker.get("formal") == 1


def test_pairwise_affinity_helpers():
    tracker = AffinityTracker()
    tracker.update_interaction("alice", "bob")
    tracker.update_interaction("bob", "alice", 2)
    tracker.update_mutual_affinity("alice", "bob", 5)
    tracker.update_mutual_affinity("alice", "charlie", -2)
    tracker.update_mutual_affinity("alice", "dave", 1)

    assert tracker.interaction_count("alice", "bob") == 3
    assert tracker.mutual_affinity("alice", "bob") == 5
    assert tracker.top_friends("alice") == ["bob", "dave", "charlie"]
    assert tracker.top_rivals("alice")[0] == "charlie"


def test_goal_scheduler_order():
    sched = GoalScheduler()
    sched.add_goal("low", priority=1)
    sched.add_goal("high", priority=5)
    sched.add_goal("mid", priority=3)
    assert sched.next_goal() == "high"
    assert sched.next_goal() == "mid"
    assert sched.next_goal() == "low"
    assert sched.next_goal() is None
