from datetime import datetime, timedelta

from deepthought.quest.templates import CooldownTracker, QuestTemplate


def test_cooldown_tracker_ready_and_mark():
    tracker = CooldownTracker()
    tmpl = QuestTemplate("T", "short", cooldown=timedelta(hours=1))
    now = datetime.utcnow()

    assert tracker.ready("user", tmpl, now=now)
    tracker.mark("user", tmpl, now=now)
    assert not tracker.ready("user", tmpl, now=now + timedelta(minutes=30))
    assert tracker.ready("user", tmpl, now=now + timedelta(hours=1, minutes=1))
