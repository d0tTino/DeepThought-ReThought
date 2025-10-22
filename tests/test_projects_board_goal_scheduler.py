from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.deepthought.goal_scheduler import GoalScheduler
from src.deepthought.plugins.projects_board import ProjectRecord, ProjectsBoard


class _FixedDateTime(datetime):
    _now: datetime = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)

    @classmethod
    def now(cls, tz=None):  # type: ignore[override]
        if tz is None:
            return cls._now
        return cls._now.astimezone(tz)


@pytest.fixture()
def fixed_now(monkeypatch):
    monkeypatch.setattr(
        "src.deepthought.plugins.projects_board.datetime",
        _FixedDateTime,
    )
    return _FixedDateTime._now


def _make_board() -> ProjectsBoard:
    board = ProjectsBoard.__new__(ProjectsBoard)
    board._scheduler = GoalScheduler()
    return board


def _make_record(**overrides):
    base = dict(
        project_id=42,
        guild_id=100,
        thread_id=200,
        name="Launch Sequence",
        summary=None,
        owner_id=None,
        status="to-do",
        due_date=datetime(2024, 1, 2, 15, 0, tzinfo=UTC),
        holiday=False,
        tags=[],
        priority=None,
        project_type=None,
        scheduled_event_id=None,
        created_at=_FixedDateTime._now,
        updated_at=_FixedDateTime._now,
        archived_at=None,
    )
    base.update(overrides)
    return ProjectRecord(**base)


def test_queue_goal_scheduler_reminder_formats_delay_and_reference(fixed_now):
    board = _make_board()
    reminder_time = fixed_now + timedelta(hours=2, minutes=5)
    record = _make_record()

    board._queue_goal_scheduler_reminder(record, reminder_time)

    queued = board._scheduler.next_goal()
    assert queued is not None
    delay_str, message = queued.split(":", 1)
    assert delay_str == str(int((reminder_time - fixed_now).total_seconds()))
    assert f"<#{record.thread_id}>" in message
    assert f"#{record.project_id}" in message
    assert record.due_date.isoformat() in message


def test_queue_goal_scheduler_reminder_now_when_past_due(fixed_now):
    board = _make_board()
    reminder_time = fixed_now - timedelta(minutes=1)
    record = _make_record(thread_id=None)

    board._queue_goal_scheduler_reminder(record, reminder_time)

    queued = board._scheduler.next_goal()
    assert queued is not None
    delay_str, message = queued.split(":", 1)
    assert delay_str == "0"
    assert f"project #{record.project_id}" in message
