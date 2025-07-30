import importlib

import pytest

from deepthought.perception import manipulative_detection as md


def test_detect_manipulation_categories():
    assert md.detect_manipulation("After all I've done for you") == "guilt_tripping"
    assert md.detect_manipulation("You'll regret this") == "threat"
    assert md.detect_manipulation("You're the best!") == "excessive_flattery"
    assert md.detect_manipulation("Just a normal message") is None
