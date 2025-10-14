import asyncio
import importlib
import sys
from datetime import UTC, datetime, timedelta
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import discord
from discord import app_commands


class StubTree:
    def add_command(self, *args, **kwargs):
        return None

    def remove_command(self, *args, **kwargs):
        return None


class StubScheduler:
    def __init__(self) -> None:
        self.add_goal = MagicMock()


@pytest.fixture(scope="module")
def projects_board_module() -> ModuleType:
    services_backup = sys.modules.get("deepthought.services")
    db_manager_backup = sys.modules.get("deepthought.services.db_manager")

    services_module = ModuleType("deepthought.services")
    db_manager_module = ModuleType("deepthought.services.db_manager")

    class DummyDBManager:  # noqa: D401 - simple stub for imports
        """Placeholder DB manager used in tests."""

    db_manager_module.DBManager = DummyDBManager
    services_module.DBManager = DummyDBManager
    services_module.db_manager = db_manager_module

    sys.modules["deepthought.services"] = services_module
    sys.modules["deepthought.services.db_manager"] = db_manager_module

    module = importlib.import_module("deepthought.plugins.projects_board")

    yield module

    if services_backup is not None:
        sys.modules["deepthought.services"] = services_backup
    else:  # pragma: no cover - cleanup when stub inserted
        sys.modules.pop("deepthought.services", None)
    if db_manager_backup is not None:
        sys.modules["deepthought.services.db_manager"] = db_manager_backup
    else:  # pragma: no cover - cleanup when stub inserted
        sys.modules.pop("deepthought.services.db_manager", None)


async def create_board(
    module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    *,
    require_events: bool = False,
):
    ProjectsBoard = module.ProjectsBoard
    loop = asyncio.get_running_loop()
    bot = SimpleNamespace(
        loop=loop,
        tree=StubTree(),
        wait_until_ready=AsyncMock(),
        add_view=MagicMock(),
        get_channel=MagicMock(return_value=None),
        fetch_channel=AsyncMock(return_value=None),
    )
    scheduler = StubScheduler()
    monkeypatch.setattr(ProjectsBoard, "_startup", AsyncMock())
    board = ProjectsBoard(
        bot,
        scheduler=scheduler,
        forum_channel_id=123,
        require_events=require_events,
    )
    board._ready.set()
    board._startup_task = None
    return board


@pytest.mark.asyncio
async def test_seed_tags_creates_missing_tags(
    projects_board_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    board = await create_board(projects_board_module, monkeypatch)
    required_tags = projects_board_module.REQUIRED_TAGS
    existing_names = {"Active", "Planning"}
    channel = SimpleNamespace(
        available_tags=[SimpleNamespace(name=name) for name in existing_names],
        create_tag=AsyncMock(side_effect=lambda **kwargs: SimpleNamespace(name=kwargs["name"])),
    )

    created = await board._ensure_tags(channel)  # type: ignore[arg-type]

    expected_missing = [name for name in required_tags if name not in existing_names]
    assert created == expected_missing
    assert channel.create_tag.await_count == len(expected_missing)
    for call, name in zip(channel.create_tag.await_args_list, expected_missing, strict=False):
        assert call.kwargs["name"] == name
        assert call.kwargs.get("emoji") == required_tags[name].get("emoji")


@pytest.mark.asyncio
async def test_create_project_uses_resolved_arguments(
    projects_board_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    board = await create_board(projects_board_module, monkeypatch)
    project_record_cls = projects_board_module.ProjectRecord
    default_status = projects_board_module.DEFAULT_STATUS

    channel = SimpleNamespace(guild=SimpleNamespace())
    board._fetch_forum_channel = AsyncMock(return_value=channel)

    due_text = "2025-12-24"
    due_date = datetime.fromisoformat(f"{due_text}T00:00:00+00:00")
    priority_choice = app_commands.Choice(name="🔥 P0", value="🔥 P0")
    project_type_choice = app_commands.Choice(name="Commission", value="Commission")

    record = project_record_cls(
        project_id=7,
        thread_id=None,
        name="Winter Launch",
        summary="Test summary",
        owner_id=None,
        status=default_status,
        due_date=due_date,
        holiday=False,
        tags=[],
        scheduled_event_id=None,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        archived_at=None,
    )

    board._create_project_record = AsyncMock(return_value=record)
    board._update_index_embed = AsyncMock()
    board._sync_due_date_reminder = AsyncMock()

    interaction = SimpleNamespace(
        response=SimpleNamespace(send_message=AsyncMock())
    )

    command = board.create_project
    await command.callback(
        board,
        interaction,
        name="Winter Launch",
        summary="Test summary",
        due_date=due_text,
        owner=None,
        tags="alpha, beta",
        holiday=False,
        priority=priority_choice,
        project_type=project_type_choice,
    )

    board._fetch_forum_channel.assert_awaited_once()
    assert board._create_project_record.await_count == 1
    kwargs = board._create_project_record.await_args.kwargs
    assert kwargs["status"] == default_status
    assert kwargs["due_date"] == due_date
    assert kwargs["priority"] == "p0"
    assert kwargs["project_type"] == "commission"
    interaction.response.send_message.assert_awaited_once()
    board._update_index_embed.assert_awaited_once_with(channel)
    board._sync_due_date_reminder.assert_awaited_once_with(record, channel.guild)


@pytest.mark.asyncio
async def test_build_board_embed_groups_records(
    projects_board_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    board = await create_board(projects_board_module, monkeypatch)
    project_record_cls = projects_board_module.ProjectRecord

    now = datetime.now(UTC)
    urgent = project_record_cls(
        project_id=1,
        thread_id=None,
        name="Urgent",
        summary=None,
        owner_id=None,
        status="in-progress",
        due_date=now + timedelta(days=1),
        holiday=False,
        tags=["🔥 P0"],
        scheduled_event_id=None,
        created_at=now - timedelta(days=5),
        updated_at=now,
        archived_at=None,
    )
    upcoming = project_record_cls(
        project_id=2,
        thread_id=None,
        name="Upcoming",
        summary=None,
        owner_id=None,
        status="in-progress",
        due_date=now + timedelta(days=14),
        holiday=False,
        tags=[],
        scheduled_event_id=None,
        created_at=now - timedelta(days=4),
        updated_at=now - timedelta(days=1),
        archived_at=None,
    )
    holiday_done = project_record_cls(
        project_id=3,
        thread_id=None,
        name="Holiday Wrap-Up",
        summary=None,
        owner_id=None,
        status="done",
        due_date=now + timedelta(days=40),
        holiday=True,
        tags=["Holiday"],
        scheduled_event_id=None,
        created_at=now - timedelta(days=10),
        updated_at=now,
        archived_at=None,
    )
    recently_done = project_record_cls(
        project_id=4,
        thread_id=None,
        name="Shipped",
        summary=None,
        owner_id=None,
        status="done",
        due_date=None,
        holiday=False,
        tags=[],
        scheduled_event_id=None,
        created_at=now - timedelta(days=8),
        updated_at=now - timedelta(days=2),
        archived_at=None,
    )

    embed = board._build_board_embed(
        [urgent, upcoming, holiday_done, recently_done],
        holiday_only=False,
    )

    fields = {field.name: field.value for field in embed.fields}
    assert {"🔥 Now", "🟠 Next", "🎁 Holiday Radar", "✅ Recently Done"} <= fields.keys()
    assert f"[# {urgent.project_id}]".replace(" ", "") in fields["🔥 Now"].replace(" ", "")
    assert upcoming.name in fields["🟠 Next"]
    assert holiday_done.name in fields["🎁 Holiday Radar"]
    assert holiday_done.name in fields["✅ Recently Done"]


@pytest.mark.asyncio
async def test_sync_due_date_reminder_falls_back_to_scheduler(
    projects_board_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    board = await create_board(projects_board_module, monkeypatch, require_events=True)
    project_record_cls = projects_board_module.ProjectRecord

    scheduler_mock = board._scheduler.add_goal
    scheduler_mock.reset_mock()

    due_date = datetime.now(UTC) + timedelta(days=2)
    record = project_record_cls(
        project_id=9,
        thread_id=None,
        name="Release",
        summary="Release prep",
        owner_id=None,
        status="in-progress",
        due_date=due_date,
        holiday=False,
        tags=[],
        scheduled_event_id=None,
        created_at=datetime.now(UTC) - timedelta(days=1),
        updated_at=datetime.now(UTC),
        archived_at=None,
    )

    guild = SimpleNamespace(
        create_scheduled_event=AsyncMock(
            side_effect=discord.Forbidden(MagicMock(), "disabled")
        ),
        get_scheduled_event=MagicMock(return_value=None),
        fetch_scheduled_event=AsyncMock(return_value=None),
    )
    board._set_scheduled_event_id = AsyncMock()

    await board._sync_due_date_reminder(record, guild)  # type: ignore[arg-type]

    guild.create_scheduled_event.assert_awaited()
    scheduler_mock.assert_called_once()
    message = scheduler_mock.call_args.args[0]
    assert record.name in message


@pytest.mark.asyncio
async def test_is_holiday_project_helper(
    projects_board_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    board = await create_board(projects_board_module, monkeypatch)

    november_due = datetime(2025, 11, 5, tzinfo=UTC)
    october_due = datetime(2025, 10, 5, tzinfo=UTC)

    assert board._is_holiday_project(november_due, []) is True
    assert board._is_holiday_project(october_due, ["Holiday"]) is True
    assert board._is_holiday_project(None, []) is False
