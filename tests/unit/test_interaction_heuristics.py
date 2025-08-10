from deepthought.bot import interaction


def test_is_bot_heuristic():
    class User:
        def __init__(self, name, bot=False):
            self.name = name
            self.bot = bot

    assert interaction.is_bot(User("HelperBot"))
    assert interaction.is_bot(User("Alice", bot=True))
    assert not interaction.is_bot(User("Alice"))


def test_is_crowded_heuristic():
    participants = ["AlphaBot", "BetaBot", "Charlie"]
    assert interaction.is_crowded(participants)
    assert not interaction.is_crowded(["Alice", "Bob"])


def test_choose_style_levels():
    assert interaction.choose_style(0.1) == "silence"
    assert interaction.choose_style(0.3) == "emoji"
    assert interaction.choose_style(0.5) == "word"
    assert interaction.choose_style(0.7) == "one-liner"
    assert interaction.choose_style(0.9) == "paragraph"


def test_choose_style_tone_adjustment():
    assert interaction.choose_style(0.5, tone="concise") == "emoji"
    assert interaction.choose_style(0.5, tone="verbose") == "one-liner"
