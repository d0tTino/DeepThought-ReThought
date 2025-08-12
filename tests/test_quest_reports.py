"""Tests for quest report generation."""

from datetime import timedelta

from deepthought.quest import (
    Epiphany,
    LieRecord,
    Objective,
    Quest,
    QuestWriter,
    SummaryScheduler,
    case_files,
    compile_narratives,
    weekly_faction_shifts,
)


def _sample_quest() -> Quest:
    quest = Quest(
        id=1,
        name="Secret Mission",
        description="Gather intel",
        faction="alpha",
        status="completed",
    )
    quest.objectives = [Objective(id=1, quest_id=1, description="Infiltrate base")]
    quest.epiphanies = [Epiphany(id=1, quest_id=1, insight="New lead")]
    quest.lies = [LieRecord(id=1, quest_id=1, lie="Covered tracks")]
    return quest


def test_compile_narratives_format():
    narrative = compile_narratives([_sample_quest()])[0]
    assert "Secret Mission" in narrative
    assert "Objectives: Infiltrate base" in narrative
    assert "Epiphanies: New lead" in narrative
    assert "Lies: Covered tracks" in narrative


def test_weekly_faction_shifts_counts_status():
    quests = [
        _sample_quest(),
        Quest(id=2, name="Failed", description="", faction="alpha", status="failed"),
        Quest(id=3, name="Beta", description="", faction="beta", status="completed"),
    ]
    shifts = weekly_faction_shifts(quests)
    assert shifts == {"alpha": 0, "beta": 1}


def test_case_files_structure():
    cases = case_files([_sample_quest()])
    assert cases[0]["name"] == "Secret Mission"
    assert cases[0]["objectives"] == ["Infiltrate base"]
    assert cases[0]["epiphanies"] == ["New lead"]
    assert cases[0]["lies"] == ["Covered tracks"]


def test_scheduler_sends_summary_when_due():
    quest = _sample_quest()
    sent = {}

    class DummyWriter:
        def send(self, summary):
            sent.update(summary)

    scheduler = SummaryScheduler(interval=timedelta(seconds=0), writer=DummyWriter())
    scheduler.maybe_send([quest])
    assert set(sent) == {"narratives", "faction_shifts", "case_files"}
    assert sent["faction_shifts"] == {"alpha": 1}


def test_send_quest_story_posts_to_autopsy(monkeypatch):
    quest = _sample_quest()
    posted = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        posted.update({"url": url, "json": json})

        class Resp:
            status_code = 200

        return Resp()

    monkeypatch.setenv("AUTOPSY_CHANNEL", "42")
    monkeypatch.setenv("DISCORD_TOKEN", "token")
    monkeypatch.setattr("requests.post", fake_post)

    writer = QuestWriter()
    compile_narratives([quest], writer=writer)

    assert posted["url"].endswith("/42/messages")
    assert "Secret Mission" in posted["json"]["content"]


def test_send_quest_story_ignores_incomplete(monkeypatch):
    quest = _sample_quest()
    quest.status = "pending"
    called = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        called["called"] = True

    monkeypatch.setenv("AUTOPSY_CHANNEL", "42")
    monkeypatch.setenv("DISCORD_TOKEN", "token")
    monkeypatch.setattr("requests.post", fake_post)

    writer = QuestWriter()
    compile_narratives([quest], writer=writer)

    assert called == {}
