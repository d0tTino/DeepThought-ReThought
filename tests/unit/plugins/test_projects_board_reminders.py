from __future__ import annotations

import importlib
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from deepthought.goal_scheduler import GoalScheduler
from deepthought.plugins.projects_board import ProjectRecord, ProjectsBoard

sys.modules.pop("examples.social_graph_bot", None)
sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))
social_graph_bot = importlib.import_module("examples.social_graph_bot")


class _FixedDateTime(datetime):
    _now = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)

    @classmethod
    def now(cls, tz=None):  # type: ignore[override]
        if tz is None:
            return cls._now
        return cls._now.astimezone(tz)


@pytest.mark.asyncio
async def test_process_goals_publishes_without_scheduler_service(monkeypatch: pytest.MonkeyPatch) -> None:
    bot = SimpleNamespace()
    bot.scheduler_service = None
    bot.goal_scheduler = GoalScheduler()
    bot._closed = False

    async def wait_until_ready() -> None:
        return None

    def is_closed() -> bool:
        return bot._closed

    bot.wait_until_ready = wait_until_ready
    bot.is_closed = is_closed

    reminder_message = "Reminder: Check project thread"
    bot.goal_scheduler.add_goal(f"0:{reminder_message}", priority=5)

    publish_mock = AsyncMock()
    monkeypatch.setattr(
        social_graph_bot,
        "publish_plan_requested",
        publish_mock,
        raising=False,
    )

    post_mock = AsyncMock()
    monkeypatch.setattr(
        social_graph_bot,
        "_maybe_post_reminder_to_thread",
        post_mock,
    )

    async def fake_sleep(_seconds: float) -> None:
        bot._closed = True
        return None

    monkeypatch.setattr(social_graph_bot.asyncio, "sleep", fake_sleep)

    await social_graph_bot.process_goals(bot)  # type: ignore[arg-type]

    publish_mock.assert_awaited_once()
    published_message = publish_mock.await_args.args[0]
    assert published_message == reminder_message
    assert bot.goal_scheduler.next_goal() is None
    post_mock.assert_awaited_once()
    assert post_mock.await_args.args[1] == reminder_message


@pytest.mark.asyncio
async def test_queue_goal_scheduler_reminder_single_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "deepthought.plugins.projects_board.datetime",
        _FixedDateTime,
    )
    board = ProjectsBoard.__new__(ProjectsBoard)
    board._scheduler = GoalScheduler()

    record = ProjectRecord(
        project_id=101,
        guild_id=555,
        thread_id=404,
        name="Regression Reminder",
        summary=None,
        owner_id=None,
        status="in-progress",
        due_date=_FixedDateTime._now + timedelta(days=1),
        holiday=False,
        tags=[],
        priority=None,
        project_type=None,
        scheduled_event_id=None,
        created_at=_FixedDateTime._now - timedelta(days=2),
        updated_at=_FixedDateTime._now - timedelta(days=1),
        archived_at=None,
    )

    board._queue_goal_scheduler_reminder(record, _FixedDateTime._now + timedelta(hours=3))

    queued = board._scheduler.next_goal()
    assert queued is not None
    assert board._scheduler.next_goal() is None

    delay_str, message = queued.split(":", 1)
    assert delay_str == str(int(timedelta(hours=3).total_seconds()))
    assert f"<#{record.thread_id}>" in message
    assert record.due_date.isoformat() in message
    assert f"ID #{record.project_id}" in message


@pytest.mark.asyncio
async def test_process_goals_posts_to_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    channel_messages: list[str] = []

    class _Channel:
        async def send(self, content: str) -> None:
            channel_messages.append(content)

    bot = SimpleNamespace()
    bot.scheduler_service = None
    bot.goal_scheduler = GoalScheduler()
    bot._closed = False
    bot.get_channel = lambda _cid: _Channel()
    bot.fetch_channel = AsyncMock(return_value=_Channel())

    async def wait_until_ready() -> None:
        return None

    def is_closed() -> bool:
        return bot._closed

    bot.wait_until_ready = wait_until_ready
    bot.is_closed = is_closed

    reminder_message = "Reminder: Visit thread <#4321>"
    bot.goal_scheduler.add_goal(f"0:{reminder_message}", priority=5)

    publish_mock = AsyncMock()
    monkeypatch.setattr(
        social_graph_bot,
        "publish_plan_requested",
        publish_mock,
        raising=False,
    )

    async def fake_sleep(_seconds: float) -> None:
        bot._closed = True
        return None

    monkeypatch.setattr(social_graph_bot.asyncio, "sleep", fake_sleep)

    await social_graph_bot.process_goals(bot)  # type: ignore[arg-type]

    publish_mock.assert_awaited_once()
    assert channel_messages == [reminder_message]
