import pytest
from datetime import datetime

from deepthought.metrics import bleu, rouge_l, average_latency, actions_per_second
from deepthought.harness.record import TraceEvent


def _event(latency: float) -> TraceEvent:
    return TraceEvent(
        state="",
        action="",
        reward=0.0,
        latency=latency,
        timestamp=datetime.utcnow(),
        timestamp_delta=0.0,
    )


def test_bleu_identical() -> None:
    text = "the cat is here"
    assert bleu(text, text) == pytest.approx(1.0)


def test_bleu_empty_candidate() -> None:
    assert bleu("", "hello world") == pytest.approx(0.0)


def test_rouge_l_identical() -> None:
    assert rouge_l("a b c", "a b c") == pytest.approx(1.0)


def test_rouge_l_no_match() -> None:
    assert rouge_l("foo", "bar") == pytest.approx(0.0)


def test_average_latency() -> None:
    events = [_event(1.0), _event(2.0)]
    assert average_latency(events) == pytest.approx(1.5)


def test_actions_per_second() -> None:
    events = [_event(1.0), _event(1.0)]
    assert actions_per_second(events) == pytest.approx(1.0)
