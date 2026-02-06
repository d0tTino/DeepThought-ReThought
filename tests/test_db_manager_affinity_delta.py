import pytest

from deepthought.services.db_manager import DBManager


def test_sentiment_to_affinity_delta_scales_and_clamps():
    assert DBManager.sentiment_to_affinity_delta(0.1) == 0
    assert DBManager.sentiment_to_affinity_delta(0.2) == 1
    assert DBManager.sentiment_to_affinity_delta(0.9) == 3
    assert DBManager.sentiment_to_affinity_delta(-0.9) == -3


def test_sentiment_to_affinity_delta_validates_input():
    with pytest.raises(ValueError, match="sentiment_score must be numeric"):
        DBManager.sentiment_to_affinity_delta("nope")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sentiment_score out of range"):
        DBManager.sentiment_to_affinity_delta(1.1)
