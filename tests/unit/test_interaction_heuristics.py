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
