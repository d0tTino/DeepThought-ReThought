from deepthought.perception.emotion_detection import detect_emotions


def test_happy_detection():
    scores = detect_emotions("I am feeling so joyful and excited today!")
    assert scores.get("Happy", 0) > scores.get("Sad", 0)


def test_sad_detection():
    scores = detect_emotions("I am feeling very sad and down right now.")
    assert scores.get("Sad", 0) > scores.get("Happy", 0)
