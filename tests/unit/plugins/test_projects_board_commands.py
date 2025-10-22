from types import SimpleNamespace
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
from discord import app_commands

from tests.unit.plugins.test_projects_board import create_board  # type: ignore

pytest_plugins = ("tests.unit.plugins.test_projects_board",)


@pytest.mark.asyncio
async def test_command_registration(projects_board_module) -> None:
    project_group = projects_board_module.ProjectsBoard.project
    status_cmd = project_group.get_command("status")
    priority_cmd = project_group.get_command("priority")
    due_cmd = project_group.get_command("due")
    tag_group = project_group.get_command("tag")

    assert status_cmd is not None
    assert priority_cmd is not None
    assert due_cmd is not None
    assert isinstance(tag_group, app_commands.Group)
    assert tag_group.get_command("add") is not None
    assert tag_group.get_command("remove") is not None


@pytest.mark.asyncio
async def test_status_command_updates_record(
    projects_board_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    board = await create_board(projects_board_module, monkeypatch)
    channel = SimpleNamespace()
    board._fetch_forum_channel = AsyncMock(return_value=channel)
    board._update_project_record = AsyncMock(
        return_value=SimpleNamespace(project_id=7)
    )
    board._update_index_embed = AsyncMock()
    interaction = SimpleNamespace(
        response=SimpleNamespace(send_message=AsyncMock())
    )
    status_choice = app_commands.Choice(name="🚧 In-Progress", value="in-progress")

    command = board.set_project_status
    await command.callback(board, interaction, project_id=7, status=status_choice)

    board._update_project_record.assert_awaited_once()
    kwargs = board._update_project_record.await_args.kwargs
    assert kwargs["status"] == "in-progress"
    interaction.response.send_message.assert_awaited_once()
    board._update_index_embed.assert_awaited_once_with(channel)


@pytest.mark.asyncio
async def test_priority_command_sets_canonical_key(
    projects_board_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    board = await create_board(projects_board_module, monkeypatch)
    channel = SimpleNamespace()
    board._fetch_forum_channel = AsyncMock(return_value=channel)
    board._set_project_priority = AsyncMock(
        return_value=SimpleNamespace(project_id=12)
    )
    board._update_index_embed = AsyncMock()
    interaction = SimpleNamespace(
        response=SimpleNamespace(send_message=AsyncMock())
    )
    priority_choice = app_commands.Choice(name="🔥 P0", value="p0")

    command = board.set_project_priority_command
    await command.callback(board, interaction, project_id=12, priority=priority_choice)

    board._set_project_priority.assert_awaited_once_with(12, "p0", channel)
    interaction.response.send_message.assert_awaited_once()
    board._update_index_embed.assert_awaited_once_with(channel)


@pytest.mark.asyncio
async def test_due_command_clears_date(
    projects_board_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    board = await create_board(projects_board_module, monkeypatch)
    channel = SimpleNamespace(guild=SimpleNamespace())
    board._fetch_forum_channel = AsyncMock(return_value=channel)
    board._update_project_record = AsyncMock(
        return_value=SimpleNamespace(project_id=3)
    )
    board._update_index_embed = AsyncMock()
    board._sync_due_date_reminder = AsyncMock()
    interaction = SimpleNamespace(
        response=SimpleNamespace(send_message=AsyncMock())
    )

    command = board.set_project_due
    await command.callback(board, interaction, project_id=3, clear=True)

    kwargs = board._update_project_record.await_args.kwargs
    assert kwargs["clear_due"] is True
    assert kwargs["due_date"] is None
    interaction.response.send_message.assert_awaited_once()
    board._update_index_embed.assert_awaited_once_with(channel)
    board._sync_due_date_reminder.assert_awaited_once()


@pytest.mark.asyncio
async def test_due_command_sets_date(
    projects_board_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    board = await create_board(projects_board_module, monkeypatch)
    channel = SimpleNamespace(guild=SimpleNamespace())
    board._fetch_forum_channel = AsyncMock(return_value=channel)
    board._update_project_record = AsyncMock(
        return_value=SimpleNamespace(project_id=9)
    )
    board._update_index_embed = AsyncMock()
    board._sync_due_date_reminder = AsyncMock()
    interaction = SimpleNamespace(
        response=SimpleNamespace(send_message=AsyncMock())
    )

    due_text = "2025-05-01"
    command = board.set_project_due
    await command.callback(board, interaction, project_id=9, due_date=due_text)

    kwargs = board._update_project_record.await_args.kwargs
    assert isinstance(kwargs["due_date"], datetime)
    assert kwargs["due_date"].date().isoformat() == due_text
    interaction.response.send_message.assert_awaited_once()
    board._update_index_embed.assert_awaited_once_with(channel)
    board._sync_due_date_reminder.assert_awaited_once()


@pytest.mark.asyncio
async def test_tag_add_uses_resolve_and_persist(
    projects_board_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    board = await create_board(projects_board_module, monkeypatch)
    ProjectRecord = projects_board_module.ProjectRecord
    record = ProjectRecord(
        project_id=21,
        guild_id=1,
        thread_id=None,
        name="Demo",
        summary=None,
        owner_id=None,
        status="to-do",
        due_date=None,
        holiday=False,
        tags=["Existing"],
        priority=None,
        project_type=None,
        scheduled_event_id=None,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        archived_at=None,
    )
    channel = SimpleNamespace(
        available_tags=[
            SimpleNamespace(name="Existing"),
            SimpleNamespace(name="Feature"),
        ]
    )
    board._fetch_forum_channel = AsyncMock(return_value=channel)
    board._fetch_project = AsyncMock(return_value=record)
    applied_tags = [SimpleNamespace(name="Existing"), SimpleNamespace(name="Feature")]
    board._resolve_tags = AsyncMock(return_value=applied_tags)
    board._persist_tag_update = AsyncMock(return_value=record)
    board._update_index_embed = AsyncMock()
    interaction = SimpleNamespace(
        response=SimpleNamespace(send_message=AsyncMock())
    )

    command = board.add_project_tags
    await command.callback(board, interaction, project_id=21, tags="Feature")

    board._resolve_tags.assert_awaited_once()
    board._persist_tag_update.assert_awaited_once_with(record, channel, applied_tags)
    interaction.response.send_message.assert_awaited_once()
    board._update_index_embed.assert_awaited_once_with(channel)


@pytest.mark.asyncio
async def test_tag_remove_filters_tags(
    projects_board_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    board = await create_board(projects_board_module, monkeypatch)
    ProjectRecord = projects_board_module.ProjectRecord
    record = ProjectRecord(
        project_id=22,
        guild_id=1,
        thread_id=None,
        name="Cleanup",
        summary=None,
        owner_id=None,
        status="to-do",
        due_date=None,
        holiday=False,
        tags=["Existing", "Feature"],
        priority=None,
        project_type=None,
        scheduled_event_id=None,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        archived_at=None,
    )
    channel = SimpleNamespace(available_tags=[])
    board._fetch_forum_channel = AsyncMock(return_value=channel)
    board._fetch_project = AsyncMock(return_value=record)
    board._resolve_tags = AsyncMock(return_value=[SimpleNamespace(name="Existing")])
    board._persist_tag_update = AsyncMock(return_value=record)
    board._update_index_embed = AsyncMock()
    interaction = SimpleNamespace(
        response=SimpleNamespace(send_message=AsyncMock())
    )

    command = board.remove_project_tags
    await command.callback(board, interaction, project_id=22, tags="Feature")

    board._resolve_tags.assert_awaited_once()
    board._persist_tag_update.assert_awaited_once()
    interaction.response.send_message.assert_awaited_once()
    board._update_index_embed.assert_awaited_once_with(channel)
