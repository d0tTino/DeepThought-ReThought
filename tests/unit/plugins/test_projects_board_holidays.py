"""Holiday tagging behaviour tests for the projects board helpers."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest_plugins = ("tests.unit.plugins.test_projects_board",)

from tests.unit.plugins.test_projects_board import create_board as create_board_helper


class DummyThread:
    def __init__(self, thread_id: int) -> None:
        self.id = thread_id
        self.add_user = AsyncMock()
        self._message = SimpleNamespace(edit=AsyncMock())
        self.last_edit_kwargs: dict[str, object] | None = None

    async def edit(self, **kwargs: object) -> None:  # pragma: no cover - simple stub
        self.last_edit_kwargs = kwargs

    async def fetch_message(self, message_id: int) -> SimpleNamespace:
        return self._message


class DummyChannel:
    def __init__(self) -> None:
        self.guild = SimpleNamespace(id=9876)
        self.available_tags: list[SimpleNamespace] = []
        self._next_thread_id = 100
        self._threads: dict[int, DummyThread] = {}

    async def create_tag(self, name: str, emoji: str | None = None) -> SimpleNamespace:
        tag = SimpleNamespace(name=name, emoji=emoji)
        self.available_tags.append(tag)
        return tag

    async def create_thread(
        self, *, name: str, content: str, applied_tags: list[SimpleNamespace]
    ) -> DummyThread:
        self._next_thread_id += 1
        thread = DummyThread(self._next_thread_id)
        self._threads[thread.id] = thread
        return thread

    def get_thread(self, thread_id: int) -> DummyThread | None:
        return self._threads.get(thread_id)

    async def fetch_thread(self, thread_id: int) -> DummyThread | None:
        return self._threads.get(thread_id)


async def _create_board(
    projects_board_module, monkeypatch: pytest.MonkeyPatch, tmp_path, *, holiday_locale: str | None
):
    db_path = tmp_path / "board.db"
    settings_stub = SimpleNamespace(social_graph_db=str(db_path))
    monkeypatch.setattr(projects_board_module, "get_settings", lambda: settings_stub)
    board = await create_board_helper(
        projects_board_module,
        monkeypatch,
        holiday_locale=holiday_locale,
    )
    await board._init_db()
    return board


@pytest.mark.asyncio
@pytest.mark.parametrize("holiday_locale", ["US", "GB"])
@pytest.mark.parametrize(
    "scenario, due_date, expected",
    [
        (
            "halloween",
            datetime(2024, 10, 5, tzinfo=UTC),
            {"US": True, "GB": False},
        ),
        (
            "thanksgiving",
            datetime(2024, 11, 27, tzinfo=UTC),
            {"US": True, "GB": False},
        ),
        (
            "thanksgiving-weekend",
            datetime(2024, 12, 1, tzinfo=UTC),
            {"US": True, "GB": True},
        ),
        (
            "christmas",
            datetime(2024, 12, 20, tzinfo=UTC),
            {"US": True, "GB": True},
        ),
        (
            "new-year",
            datetime(2025, 1, 5, tzinfo=UTC),
            {"US": True, "GB": True},
        ),
        (
            "valentines",
            datetime(2025, 2, 14, tzinfo=UTC),
            {"US": True, "GB": False},
        ),
        (
            "mid-summer",
            datetime(2024, 7, 15, tzinfo=UTC),
            {"US": False, "GB": False},
        ),
    ],
)
async def test_creation_detects_holiday_windows(
    tmp_path,
    projects_board_module,
    monkeypatch: pytest.MonkeyPatch,
    holiday_locale: str,
    scenario: str,
    due_date: datetime,
    expected: dict[str, bool],
) -> None:
    board = await _create_board(projects_board_module, monkeypatch, tmp_path, holiday_locale=holiday_locale)
    channel = DummyChannel()
    try:
        record = await board._create_project_record(
            channel=channel,
            name=f"{scenario} drop",
            summary="",
            status="in-progress",
            due_date=due_date,
            owner=None,
            tags=None,
            holiday=False,
            priority="p1",
            project_type=None,
        )
    finally:
        await board._close_db()
    assert record.holiday is expected[holiday_locale]
    tag_names = set(record.tags)
    holiday_tag = projects_board_module.HOLIDAY_TAG_NAME
    assert (holiday_tag in tag_names) is expected[holiday_locale]


@pytest.mark.asyncio
@pytest.mark.parametrize("holiday_locale", ["US", "GB"])
async def test_creation_respects_manual_holiday_override(
    tmp_path,
    projects_board_module,
    monkeypatch: pytest.MonkeyPatch,
    holiday_locale: str,
) -> None:
    board = await _create_board(projects_board_module, monkeypatch, tmp_path, holiday_locale=holiday_locale)
    channel = DummyChannel()
    try:
        record = await board._create_project_record(
            channel=channel,
            name="Custom Seasonal",
            summary="",
            status="in-progress",
            due_date=datetime(2024, 6, 1, tzinfo=UTC),
            owner=None,
            tags=None,
            holiday=True,
            priority=None,
            project_type=None,
        )
    finally:
        await board._close_db()
    assert record.holiday is True
    assert projects_board_module.HOLIDAY_TAG_NAME in record.tags


@pytest.mark.asyncio
async def test_update_allows_holiday_tag_removal_outside_window(
    tmp_path,
    projects_board_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    board = await _create_board(projects_board_module, monkeypatch, tmp_path, holiday_locale="US")
    channel = DummyChannel()
    try:
        record = await board._create_project_record(
            channel=channel,
            name="Winter Blitz",
            summary="",
            status="in-progress",
            due_date=datetime(2024, 12, 22, tzinfo=UTC),
            owner=None,
            tags=None,
            holiday=False,
            priority=None,
            project_type=None,
        )
        assert record.holiday is True
        updated = await board._update_project_record(
            record.project_id,
            channel=channel,
            name=None,
            summary=None,
            status=None,
            due_date=datetime(2025, 5, 1, tzinfo=UTC),
            owner=None,
            tags=None,
            holiday=False,
            clear_due=False,
            priority=None,
            project_type=None,
        )
    finally:
        await board._close_db()
    assert updated is not None
    assert updated.holiday is False
    assert projects_board_module.HOLIDAY_TAG_NAME not in updated.tags
