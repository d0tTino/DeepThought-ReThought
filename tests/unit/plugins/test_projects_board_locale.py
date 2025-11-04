import pytest
from datetime import UTC, datetime

from tests.unit.plugins.test_projects_board import (
    create_board,
    projects_board_module,
)


@pytest.mark.asyncio
async def test_projects_board_uses_us_calendar_by_default(
    projects_board_module, monkeypatch
) -> None:
    board = await create_board(projects_board_module, monkeypatch)
    due_date = datetime(2025, 10, 31, tzinfo=UTC)

    assert board._holiday_calendar.locale == "US"
    assert board._is_holiday_project(due_date, [])


@pytest.mark.asyncio
async def test_projects_board_supports_alternate_locale(
    projects_board_module, monkeypatch
) -> None:
    board = await create_board(
        projects_board_module, monkeypatch, holiday_locale="gb"
    )
    due_date = datetime(2025, 12, 25, tzinfo=UTC)

    assert board._holiday_calendar.locale == "GB"
    assert board._is_holiday_project(due_date, [])
    assert board._is_holiday_project(None, ["HoLiDaY"])
