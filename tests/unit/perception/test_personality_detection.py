from deepthought.perception.personality_detection import infer_personality


def test_keyword_scoring():
    messages = [
        "I am very organized and always plan ahead.",
        "People say I'm helpful and cooperative.",
    ]
    scores = infer_personality(messages)
    assert scores["conscientiousness"] == 0.5
    assert scores["agreeableness"] == 0.5
    assert scores["extraversion"] == 0.0


def test_neutral_fallback():
    scores = infer_personality([])
    assert all(score == 0.5 for score in scores.values())
