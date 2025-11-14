import asyncio
import importlib
import re
import sys
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
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
    holiday_locale: str | None = None,
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
    kwargs: dict[str, object] = {}
    if holiday_locale is not None:
        kwargs["holiday_locale"] = holiday_locale
    board = ProjectsBoard(
        bot,
        scheduler=scheduler,
        forum_channel_id=123,
        require_events=require_events,
        **kwargs,
    )
    board._ready.set()
    board._startup_task = None
    return board


@pytest.mark.asyncio
async def test_seed_tags_creates_missing_tags(
    projects_board_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    board = await create_board(projects_board_module, monkeypatch)
    holiday_tag = projects_board_module.HOLIDAY_TAG_NAME
    required_tags = projects_board_module.REQUIRED_TAGS
    expected_taxonomy = {
        "⏳ To-Do",
        "🚧 In-Progress",
        "🧱 Blocked",
        "💤 On-Hold",
        "✅ Done",
        "📦 Archived",
        "🔥 P0",
        "🟠 P1",
        "🟢 P2",
        "💼 Commission",
        "🎨 Personal",
        "🤝 Collaboration",
        "🌱 Community",
        "🏢 Internal",
        "🧪 Study",
        holiday_tag,
    }
    assert set(required_tags) == expected_taxonomy
    existing_names = {"🚧 In-Progress", "⏳ To-Do"}
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
    priority_choice = app_commands.Choice(name="🔥 P0", value="p0")
    project_type_choice = app_commands.Choice(name="💼 Commission", value="💼 Commission")

    record = project_record_cls(
        project_id=7,
        guild_id=123,
        thread_id=None,
        name="Winter Launch",
        summary="Test summary",
        owner_id=None,
        status=default_status,
        due_date=due_date,
        holiday=False,
        tags=[],
        priority="p0",
        project_type="commission",
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
        guild_id=42,
        thread_id=None,
        name="Urgent",
        summary=None,
        owner_id=None,
        status="in-progress",
        due_date=now + timedelta(days=1),
        holiday=False,
        tags=["🔥 P0"],
        priority="p0",
        project_type=None,
        scheduled_event_id=None,
        created_at=now - timedelta(days=5),
        updated_at=now,
        archived_at=None,
    )
    upcoming = project_record_cls(
        project_id=2,
        guild_id=42,
        thread_id=None,
        name="Upcoming",
        summary=None,
        owner_id=None,
        status="in-progress",
        due_date=now + timedelta(days=14),
        holiday=False,
        tags=[],
        priority="p1",
        project_type=None,
        scheduled_event_id=None,
        created_at=now - timedelta(days=4),
        updated_at=now - timedelta(days=1),
        archived_at=None,
    )
    holiday_done = project_record_cls(
        project_id=3,
        guild_id=42,
        thread_id=None,
        name="Holiday Wrap-Up",
        summary=None,
        owner_id=None,
        status="done",
        due_date=now + timedelta(days=40),
        holiday=True,
        tags=["🎁 Holiday"],
        priority=None,
        project_type="holiday",
        scheduled_event_id=None,
        created_at=now - timedelta(days=10),
        updated_at=now,
        archived_at=None,
    )
    recently_done = project_record_cls(
        project_id=4,
        guild_id=42,
        thread_id=None,
        name="Shipped",
        summary=None,
        owner_id=None,
        status="done",
        due_date=None,
        holiday=False,
        tags=[],
        priority=None,
        project_type=None,
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
    assert {"🔥 P0", "🟠 P1", "🎁 Holiday Radar", "✅ Recently Done"} <= fields.keys()
    assert f"[# {urgent.project_id}]".replace(" ", "") in fields["🔥 P0"].replace(" ", "")
    assert upcoming.name in fields["🟠 P1"]
    assert holiday_done.name in fields["🎁 Holiday Radar"]
    assert holiday_done.name in fields["✅ Recently Done"]


@pytest.mark.asyncio
async def test_build_board_embed_treats_canonical_p0_as_now(
    projects_board_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    board = await create_board(projects_board_module, monkeypatch)
    project_record_cls = projects_board_module.ProjectRecord

    now = datetime.now(UTC)
    high_priority = project_record_cls(
        project_id=10,
        guild_id=43,
        thread_id=None,
        name="Critical Initiative",
        summary=None,
        owner_id=None,
        status="in-progress",
        due_date=now + timedelta(days=45),
        holiday=False,
        tags=["🔥 P0"],
        priority="p0",
        project_type=None,
        scheduled_event_id=None,
        created_at=now - timedelta(days=3),
        updated_at=now,
        archived_at=None,
    )

    embed = board._build_board_embed([high_priority], holiday_only=False)

    fields = {field.name: field.value for field in embed.fields}
    assert high_priority.name in fields["🔥 P0"]
    assert fields["🟠 P1"] == "_No upcoming projects in the queue._"


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
        guild_id=44,
        thread_id=None,
        name="Release",
        summary="Release prep",
        owner_id=None,
        status="in-progress",
        due_date=due_date,
        holiday=False,
        tags=[],
        priority=None,
        project_type=None,
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
    assert scheduler_mock.call_count == 2
    first_call, second_call = scheduler_mock.call_args_list
    first_message = first_call.args[0]
    second_message = second_call.args[0]
    assert record.name in first_message
    assert record.name in second_message
    assert first_call.kwargs.get("priority") == 5
    assert second_call.kwargs.get("priority") == 5


@pytest.mark.asyncio
async def test_queue_goal_scheduler_reminder_formats_goal_for_scheduler(
    projects_board_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    board = await create_board(projects_board_module, monkeypatch)
    project_record_cls = projects_board_module.ProjectRecord

    now = datetime.now(UTC)
    record = project_record_cls(
        project_id=101,
        guild_id=55,
        thread_id=None,
        name="Scheduler Check",
        summary="Ensure reminders are queued correctly",
        owner_id=None,
        status="in-progress",
        due_date=now + timedelta(days=3),
        holiday=False,
        tags=[],
        priority=None,
        project_type=None,
        scheduled_event_id=None,
        created_at=now - timedelta(days=2),
        updated_at=now,
        archived_at=None,
    )

    scheduler_mock = board._scheduler.add_goal
    scheduler_mock.reset_mock()

    reminder_time = now + timedelta(minutes=45)
    board._queue_goal_scheduler_reminder(record, reminder_time)

    assert scheduler_mock.call_count == 2
    goal_argument = scheduler_mock.call_args_list[0].args[0]
    priority = scheduler_mock.call_args_list[0].kwargs["priority"]
    follow_up_argument = scheduler_mock.call_args_list[1].args[0]
    follow_up_priority = scheduler_mock.call_args_list[1].kwargs["priority"]
    assert priority == 5
    assert follow_up_priority == 5

    match = re.match(r"^(\d+):(.*)$", goal_argument)
    assert match is not None
    delay = int(match.group(1))
    assert delay >= 45 * 60
    message = match.group(2)
    assert record.name in message
    assert reminder_time.isoformat() in message
    assert record.name in follow_up_argument

    scheduler_mock.reset_mock()

    past_time = now - timedelta(minutes=5)
    board._queue_goal_scheduler_reminder(record, past_time)

    assert scheduler_mock.call_count == 2
    for call in scheduler_mock.call_args_list:
        past_goal_argument = call.args[0]
        past_match = re.match(r"^(\d+):(.*)$", past_goal_argument)
        assert past_match is not None
        assert int(past_match.group(1)) == 0
        assert record.name in past_match.group(2)
        assert call.kwargs.get("priority") == 5


@pytest.mark.asyncio
async def test_create_or_update_event_uses_external_entity_type(
    projects_board_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    board = await create_board(projects_board_module, monkeypatch, require_events=True)
    project_record_cls = projects_board_module.ProjectRecord

    due_date = datetime.now(UTC) + timedelta(days=2)
    start_time = due_date - projects_board_module.DEFAULT_REMINDER_LEAD

    record = project_record_cls(
        project_id=11,
        guild_id=44,
        thread_id=None,
        name="Deadline",
        summary="Finish milestone",
        owner_id=None,
        status="in-progress",
        due_date=due_date,
        holiday=False,
        tags=[],
        priority=None,
        project_type=None,
        scheduled_event_id=None,
        created_at=datetime.now(UTC) - timedelta(days=1),
        updated_at=datetime.now(UTC),
        archived_at=None,
    )

    event = SimpleNamespace(id=9876)
    guild = SimpleNamespace(
        create_scheduled_event=AsyncMock(return_value=event),
        get_scheduled_event=MagicMock(return_value=None),
        fetch_scheduled_event=AsyncMock(return_value=None),
    )
    board._set_scheduled_event_id = AsyncMock()

    await board._create_or_update_event(record, guild, start_time)  # type: ignore[arg-type]

    guild.create_scheduled_event.assert_awaited_once()
    kwargs = guild.create_scheduled_event.await_args.kwargs
    assert kwargs["entity_type"] is discord.EntityType.external
    assert kwargs["location"]


@pytest.mark.asyncio
async def test_is_holiday_project_helper(
    projects_board_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    board = await create_board(projects_board_module, monkeypatch)

    november_due = datetime(2025, 11, 27, tzinfo=UTC)
    october_due = datetime(2025, 10, 5, tzinfo=UTC)

    assert board._is_holiday_project(november_due, []) is True
    assert board._is_holiday_project(october_due, ["🎁 Holiday"]) is True
    assert board._is_holiday_project(None, []) is False


@pytest.mark.asyncio
async def test_project_persistence_tracks_priority_type_and_guild(
    tmp_path: Path,
    projects_board_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "board.db"
    settings_stub = SimpleNamespace(social_graph_db=str(db_path))
    monkeypatch.setattr(projects_board_module, "get_settings", lambda: settings_stub)

    board = await create_board(projects_board_module, monkeypatch)
    await board._init_db()

    class DummyChannel:
        def __init__(self, guild_id: int) -> None:
            self.guild = SimpleNamespace(id=guild_id)
            self.available_tags: list[SimpleNamespace] = []
            self._thread_id = guild_id * 100

        async def create_tag(self, name: str, emoji: str | None = None) -> SimpleNamespace:
            tag = SimpleNamespace(name=name, emoji=emoji)
            self.available_tags.append(tag)
            return tag

        async def create_thread(
            self, *, name: str, content: str, applied_tags: list[SimpleNamespace]
        ) -> SimpleNamespace:
            self._thread_id += 1
            return SimpleNamespace(id=self._thread_id, add_user=AsyncMock())

        def get_thread(self, thread_id: int) -> None:
            return None

        async def fetch_thread(self, thread_id: int) -> None:
            return None

    primary_channel = DummyChannel(777)
    secondary_channel = DummyChannel(888)

    record_later = await board._create_project_record(
        channel=primary_channel,
        name="Later Work",
        summary="",
        status="in-progress",
        due_date=None,
        owner=None,
        tags=None,
        holiday=False,
        priority="p2",
        project_type="personal",
    )
    record_now = await board._create_project_record(
        channel=primary_channel,
        name="Critical Work",
        summary="",
        status="in-progress",
        due_date=None,
        owner=None,
        tags=None,
        holiday=False,
        priority="p0",
        project_type="commission",
    )
    record_other = await board._create_project_record(
        channel=secondary_channel,
        name="Other Guild",
        summary="",
        status="in-progress",
        due_date=None,
        owner=None,
        tags=None,
        holiday=False,
        priority="p1",
        project_type="collaboration",
    )

    assert record_later.guild_id == primary_channel.guild.id
    assert record_later.priority == "p2"
    assert record_later.project_type == "personal"
    assert "🟢 P2" in record_later.tags
    assert "🎨 Personal" in record_later.tags

    assert record_now.guild_id == primary_channel.guild.id
    assert record_now.priority == "p0"
    assert record_now.project_type == "commission"
    assert "🔥 P0" in record_now.tags
    assert "💼 Commission" in record_now.tags

    assert record_other.guild_id == secondary_channel.guild.id
    assert record_other.priority == "p1"
    assert record_other.project_type == "collaboration"

    primary_records = await board._fetch_projects(guild_id=primary_channel.guild.id)
    assert [proj.project_id for proj in primary_records] == [
        record_now.project_id,
        record_later.project_id,
    ]

    secondary_records = await board._fetch_projects(guild_id=secondary_channel.guild.id)
    assert [proj.project_id for proj in secondary_records] == [record_other.project_id]

    updated_priority = await board._set_project_priority(
        record_later.project_id, "p0", primary_channel
    )
    assert updated_priority is not None
    assert updated_priority.priority == "p0"
    assert "🔥 P0" in updated_priority.tags

    updated_type = await board._set_project_type(
        record_now.project_id, "collaboration", primary_channel
    )
    assert updated_type is not None
    assert updated_type.project_type == "collaboration"
    assert "🤝 Collaboration" in updated_type.tags

    await board._close_db()


@pytest.mark.asyncio
async def test_backfill_project_metadata_normalises_priority_tags(
    tmp_path: Path,
    projects_board_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "board.db"
    settings_stub = SimpleNamespace(social_graph_db=str(db_path))
    monkeypatch.setattr(projects_board_module, "get_settings", lambda: settings_stub)

    board = await create_board(projects_board_module, monkeypatch)
    await board._init_db()

    assert board._conn is not None
    now_iso = datetime.now(UTC).isoformat()
    legacy_tags = json.dumps(["🔥 Now", "Feature"])
    await board._conn.execute(
        """
        INSERT INTO projects (guild_id, name, status, holiday, tags, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (123, "Legacy Priority", "to-do", 0, legacy_tags, now_iso, now_iso),
    )
    await board._conn.commit()

    await board._backfill_project_metadata()

    cursor = await board._conn.execute(
        "SELECT priority, tags FROM projects WHERE name = ?", ("Legacy Priority",)
    )
    row = await cursor.fetchone()
    await cursor.close()

    assert row is not None
    assert row["priority"] == "p0"
    updated_tags = json.loads(row["tags"])
    assert updated_tags[-1] == "🔥 P0"
    assert "Feature" in updated_tags

    await board._close_db()
