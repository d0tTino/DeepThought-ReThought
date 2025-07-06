import pytest

from deepthought.learn.online_lora import OnlineLoRALearner


@pytest.fixture()
def learner() -> OnlineLoRALearner:
    """Provide a fresh learner for each test."""
    return OnlineLoRALearner()


def test_record_interaction_isolation(learner: OnlineLoRALearner) -> None:
    learner.record_interaction("guild1", "p1", "r1", 1.0)
    learner.record_interaction("guild2", "p2", "r2", 2.0)

    data1 = learner.get_training_data("guild1")
    data2 = learner.get_training_data("guild2")

    assert len(data1) == 1
    assert len(data2) == 1
    assert data1[0].prompt == "p1"
    assert data2[0].prompt == "p2"
    assert data1[0].response == "r1"
    assert data2[0].response == "r2"
    assert data1[0].reward == 1.0
    assert data2[0].reward == 2.0


def test_get_training_data_returns_copy(learner: OnlineLoRALearner) -> None:
    learner.record_interaction("guild", "hi", "hello", 1.0)

    first = learner.get_training_data("guild")
    second = learner.get_training_data("guild")
    assert first == second
    assert first is not second

    first.pop()

    # Original data should remain intact
    fresh = learner.get_training_data("guild")
    assert len(fresh) == 1
    assert fresh[0].prompt == "hi"


def test_clear_data_removes_entries(learner: OnlineLoRALearner) -> None:
    learner.record_interaction("guild", "hi", "hello", 1.0)

    learner.clear_data("guild")

    assert learner.get_training_data("guild") == []

    learner.record_interaction("guild", "new", "n", 0.0)
    assert len(learner.get_training_data("guild")) == 1
