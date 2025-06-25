from deepthought.motivate.caption import summarise_message
from deepthought.motivate.scorer import score_caption


def test_caption_and_score():
    caption = summarise_message("one two three four five six", max_words=3)
    assert caption == "one two three"
    score = score_caption(caption, "nonce")
    assert 1 <= score <= 7
