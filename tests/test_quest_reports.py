"""Tests for quest report generation."""

from datetime import timedelta

from deepthought.quest import (
    Epiphany,
    Evidence,
    LieRecord,
    Objective,
    Quest,
    QuestWriter,
    SummaryScheduler,
    case_files,
    compile_narratives,
    generate_living_report,
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
    objective = Objective(id=1, quest_id=1, description="Infiltrate base")
    objective.evidence = [Evidence(id=1, objective_id=1, content="Photos", who="Agent A")]
    quest.objectives = [objective]
    quest.epiphanies = [Epiphany(id=1, quest_id=1, insight="New lead", who="Agent A")]
    quest.lies = [LieRecord(id=1, quest_id=1, lie="Covered tracks", who="Spy B")]
    return quest


def test_compile_narratives_format():
    narrative = compile_narratives([_sample_quest()])[0]
    assert "Secret Mission" in narrative
    assert "Objectives: Infiltrate base" in narrative
    assert "Cast: Agent A, Spy B" in narrative
    assert "Evidence: Photos" in narrative
    assert "Twists: Covered tracks" in narrative
    assert "Lessons: New lead" in narrative


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


def test_generate_living_report_structure():
    quest = _sample_quest()
    report = generate_living_report([quest], {"ops": 3})
    assert "weekly_arcs" in report
    assert "channel_heatmap" in report
    assert report["channel_heatmap"] == {"ops": 3}
    assert any("Secret Mission" in arc["summary"] for arc in report["weekly_arcs"])


def test_send_living_report_posts(monkeypatch):
    quest = _sample_quest()
    posted = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        posted.update({"url": url, "json": json})

        class Resp:
            status_code = 200

        return Resp()

    monkeypatch.setenv("LIVING_REPORT_CHANNEL", "99")
    monkeypatch.setenv("DISCORD_TOKEN", "token")
    monkeypatch.setattr("requests.post", fake_post)

    writer = QuestWriter()
    generate_living_report([quest], {"ops": 1}, writer=writer)

    assert posted["url"].endswith("/99/messages")
