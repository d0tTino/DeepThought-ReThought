from deepthought.perception.manipulative_detection import detect_manipulation
from deepthought.perception.manipulative_phrases import CATEGORY_PHRASES
from deepthought.services.manipulative_detection import manipulation_score


def test_each_category_detected():
    for category, phrases in CATEGORY_PHRASES.items():
        for phrase in phrases:
            assert detect_manipulation(phrase) == category
            assert manipulation_score(phrase) == category


def test_no_category_detected():
    text = "just a normal message"
    assert detect_manipulation(text) is None
    assert manipulation_score(text) is None
