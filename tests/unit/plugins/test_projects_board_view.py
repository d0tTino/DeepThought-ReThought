from datetime import datetime, timezone
from typing import Type, TypeVar
from unittest.mock import AsyncMock

import discord
from discord.ui import item as ui_item
import pytest

from deepthought.plugins.projects_board import (
    BOARD_STATUS_ORDER,
    ProjectRecord,
    ProjectsBoardView,
)


if isinstance(ui_item.Item.view, property) and ui_item.Item.view.fset is None:
    ui_item.Item.view = ui_item.Item.view.setter(lambda self, value: setattr(self, "_view", value))

if isinstance(discord.ui.Select.values, property) and discord.ui.Select.values.fset is None:
    discord.ui.Select.values = discord.ui.Select.values.setter(
        lambda self, value: setattr(self, "_values", value)
    )


class _FakeInteractionResponse:
    def __init__(self) -> None:
        self.send_message = AsyncMock()


class _FakeInteraction:
    def __init__(self) -> None:
        self.response = _FakeInteractionResponse()


class StubBoard:
    def __init__(self) -> None:
        self._handle_project_selection = AsyncMock()
        self._handle_action_selection = AsyncMock()
        self._handle_refresh = AsyncMock()
        self._handle_toggle_holiday = AsyncMock()
        self._handle_clear_selection = AsyncMock()

    def _format_due_label(self, due_date: datetime | None) -> str:
        return "due soon" if due_date else "no due date"

    def _status_display_name(self, status_key: str) -> str:
        return f"status:{status_key}"

    def _normalise_status_key(self, status: str) -> str:
        return status

    def _priority_for_record(self, record: ProjectRecord) -> str | None:
        return record.priority

    def _project_type_for_record(self, record: ProjectRecord) -> str | None:
        return record.project_type


def _make_record(
    *,
    project_id: int,
    name: str,
    status: str,
    due_date: datetime | None,
    priority: str | None,
    project_type: str | None,
) -> ProjectRecord:
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return ProjectRecord(
        project_id=project_id,
        guild_id=1,
        thread_id=None,
        name=name,
        summary=None,
        owner_id=None,
        status=status,
        due_date=due_date,
        holiday=False,
        tags=[],
        priority=priority,
        project_type=project_type,
        scheduled_event_id=None,
        created_at=now,
        updated_at=now,
        archived_at=None,
    )


TView = TypeVar("TView", bound=ProjectsBoardView)


def _get_child(view: ProjectsBoardView, child_type: Type[TView]) -> TView:
    for child in view.children:
        if isinstance(child, child_type):
            return child
    raise AssertionError(f"Expected to find child of type {child_type.__name__}")


@pytest.mark.asyncio()
async def test_projects_board_view_populates_options_and_defaults() -> None:
    board = StubBoard()
    records = [
        _make_record(
            project_id=1,
            name="Alpha",
            status="to-do",
            due_date=datetime(2024, 5, 1, tzinfo=timezone.utc),
            priority="p0",
            project_type="commission",
        ),
        _make_record(
            project_id=2,
            name="Beta",
            status="blocked",
            due_date=None,
            priority="p1",
            project_type="collaboration",
        ),
    ]

    view = ProjectsBoardView(
        board,
        board_id=7,
        records=records,
        selected_project_id=2,
        holiday_only=False,
    )

    project_select = _get_child(view, ProjectsBoardView.ProjectSelect)
    assert project_select.disabled is False
    assert [option.value for option in project_select.options] == ["1", "2"]
    assert project_select.options[0].label == "#1 · Alpha"
    assert project_select.options[1].default is True
    assert project_select.options[0].description == "status:to-do · due soon"

    action_select = _get_child(view, ProjectsBoardView.ActionSelect)
    assert action_select.disabled is False

    status_values = {option.value for option in action_select.options if option.value.startswith("status:")}
    assert status_values == {f"status:{status}" for status in BOARD_STATUS_ORDER}

    blocked_option = next(option for option in action_select.options if option.value == "status:blocked")
    assert blocked_option.default is True

    priority_option = next(option for option in action_select.options if option.value == "priority:p1")
    assert priority_option.default is True

    clear_priority = next(option for option in action_select.options if option.value == "priority:clear")
    assert clear_priority.default is False

    type_option = next(option for option in action_select.options if option.value == "type:collaboration")
    assert type_option.default is True

    clear_type = next(option for option in action_select.options if option.value == "type:clear")
    assert clear_type.default is False

    archive_option = next(option for option in action_select.options if option.value == "archive")
    assert archive_option.default is False


@pytest.mark.asyncio()
async def test_projects_board_view_dispatches_handlers_for_selects() -> None:
    board = StubBoard()
    records = [
        _make_record(
            project_id=1,
            name="Alpha",
            status="to-do",
            due_date=None,
            priority="p0",
            project_type="commission",
        ),
        _make_record(
            project_id=2,
            name="Beta",
            status="blocked",
            due_date=None,
            priority="p1",
            project_type="collaboration",
        ),
    ]
    view = ProjectsBoardView(
        board,
        board_id=7,
        records=records,
        selected_project_id=2,
        holiday_only=False,
    )

    project_select = _get_child(view, ProjectsBoardView.ProjectSelect)
    project_select.values = ["1"]
    interaction = _FakeInteraction()
    await project_select.callback(interaction)
    board._handle_project_selection.assert_awaited_once_with(interaction, 7, 1)

    action_select = _get_child(view, ProjectsBoardView.ActionSelect)
    for value in ["status:to-do", "priority:clear", "archive"]:
        board._handle_action_selection.reset_mock()
        action_select.values = [value]
        await action_select.callback(interaction)
        board._handle_action_selection.assert_awaited_once_with(
            interaction,
            7,
            2,
            value,
        )


@pytest.mark.asyncio()
async def test_projects_board_view_buttons_dispatch_to_cog_handlers() -> None:
    board = StubBoard()
    record = _make_record(
        project_id=1,
        name="Solo",
        status="to-do",
        due_date=None,
        priority="p0",
        project_type="personal",
    )
    view = ProjectsBoardView(
        board,
        board_id=5,
        records=[record],
        selected_project_id=1,
        holiday_only=False,
    )
    interaction = _FakeInteraction()

    refresh_button = _get_child(view, ProjectsBoardView.RefreshButton)
    await refresh_button.callback(interaction)
    board._handle_refresh.assert_awaited_once_with(interaction, 5)

    holiday_button = _get_child(view, ProjectsBoardView.HolidayButton)
    await holiday_button.callback(interaction)
    board._handle_toggle_holiday.assert_awaited_once_with(interaction, 5)

    clear_button = _get_child(view, ProjectsBoardView.ClearSelectionButton)
    await clear_button.callback(interaction)
    board._handle_clear_selection.assert_awaited_once_with(interaction, 5)


@pytest.mark.asyncio()
async def test_projects_board_view_handles_empty_project_list() -> None:
    board = StubBoard()
    view = ProjectsBoardView(
        board,
        board_id=3,
        records=[],
        selected_project_id=None,
        holiday_only=False,
    )

    project_select = _get_child(view, ProjectsBoardView.ProjectSelect)
    assert project_select.disabled is True
    assert project_select.options[0].value == "noop"
    assert project_select.options[0].label == "No projects available"

    action_select = _get_child(view, ProjectsBoardView.ActionSelect)
    assert action_select.disabled is True
    assert action_select.options[0].value == "noop"
    assert action_select.options[0].default is True


@pytest.mark.asyncio()
async def test_projects_board_view_holiday_mode_updates_button_label_and_style() -> None:
    board = StubBoard()
    record = _make_record(
        project_id=1,
        name="Gamma",
        status="done",
        due_date=datetime(2024, 4, 1, tzinfo=timezone.utc),
        priority=None,
        project_type=None,
    )
    view = ProjectsBoardView(
        board,
        board_id=4,
        records=[record],
        selected_project_id=1,
        holiday_only=True,
    )

    holiday_button = _get_child(view, ProjectsBoardView.HolidayButton)
    assert holiday_button.label == "Holiday Filter: On"
    assert holiday_button.style is discord.ButtonStyle.success
