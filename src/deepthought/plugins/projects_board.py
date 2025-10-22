"""Discord projects board management cog."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import os
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any, Iterable, List

import aiosqlite
import discord
from discord import app_commands
from discord.ext import commands

from ..config import get_settings
from ..goal_scheduler import GoalScheduler
from ..services.db_manager import DBManager

__all__ = ["ProjectsBoard", "ProjectRecord"]


_LOG = logging.getLogger(__name__)

INDEX_THREAD_NAME = "Projects Index"
DEFAULT_STATUS = "to-do"
DEFAULT_REMINDER_LEAD = timedelta(hours=1)
DEFAULT_EVENT_LOCATION = "Projects board reminder"
SCHEDULED_EVENT_LOCATION = "Projects Board Reminder"


def _normalize_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _is_halloween_window(moment: datetime) -> bool:
    day = _normalize_datetime(moment).date()
    return day.month == 10


def _fourth_thursday(year: int) -> date:
    first_day = date(year, 11, 1)
    offset = (3 - first_day.weekday()) % 7
    return first_day + timedelta(days=offset + (4 - 1) * 7)


def _is_thanksgiving_window(moment: datetime) -> bool:
    day = _normalize_datetime(moment).date()
    if day.month != 11:
        return False
    thanksgiving = _fourth_thursday(day.year)
    window_start = thanksgiving - timedelta(days=3)
    window_end = thanksgiving + timedelta(days=2)
    return window_start <= day <= window_end


def _is_christmas_new_year_window(moment: datetime) -> bool:
    day = _normalize_datetime(moment).date()
    if day.month == 12:
        return True
    if day.month == 1 and day.day <= 7:
        return True
    return False


def _is_valentines_window(moment: datetime) -> bool:
    day = _normalize_datetime(moment).date()
    return day.month == 2 and 1 <= day.day <= 21

BOARD_STATUS_META: dict[str, dict[str, str]] = {
    "to-do": {"label": "⏳ To-Do", "tag": "⏳ To-Do"},
    "in-progress": {"label": "🚧 In-Progress", "tag": "🚧 In-Progress"},
    "blocked": {"label": "🧱 Blocked", "tag": "🧱 Blocked"},
    "on-hold": {"label": "💤 On-Hold", "tag": "💤 On-Hold"},
    "done": {"label": "✅ Done", "tag": "✅ Done"},
    "archived": {"label": "📦 Archived", "tag": "📦 Archived"},
}

BOARD_STATUS_ORDER: list[str] = [
    "to-do",
    "in-progress",
    "blocked",
    "on-hold",
    "done",
]

HOLIDAY_TAG_NAME = "🎁 Holiday"
HOLIDAY_TAG_ALIASES = {HOLIDAY_TAG_NAME.casefold(), "holiday"}

REQUIRED_TAGS: dict[str, dict[str, str | None]] = {
    "⏳ To-Do": {"emoji": "⏳"},
    "🚧 In-Progress": {"emoji": "🚧"},
    "🧱 Blocked": {"emoji": "🧱"},
    "💤 On-Hold": {"emoji": "💤"},
    "✅ Done": {"emoji": "✅"},
    "📦 Archived": {"emoji": "📦"},
    HOLIDAY_TAG_NAME: {"emoji": "🎁"},
}

STATUS_TAGS: dict[str, str] = {
    key: meta["tag"] for key, meta in BOARD_STATUS_META.items()
}
for meta in BOARD_STATUS_META.values():
    label = meta["label"]
    tag = meta["tag"]
    STATUS_TAGS[label.casefold()] = tag
    STATUS_TAGS[tag.casefold()] = tag
    text = label.split(" ", 1)[-1]
    STATUS_TAGS[text.casefold()] = tag
    STATUS_TAGS[text.replace("-", " ").casefold()] = tag

STATUS_TAGS.update(
    {
        "active": "🚧 In-Progress",
        "planning": "⏳ To-Do",
        "completed": "✅ Done",
        "complete": "✅ Done",
        "todo": "⏳ To-Do",
        "to do": "⏳ To-Do",
        "in progress": "🚧 In-Progress",
        "on hold": "💤 On-Hold",
        "blocked": "🧱 Blocked",
        "archived": "📦 Archived",
    }
)

LEGACY_STATUS_TAG_NAMES: dict[str, tuple[str, ...]] = {
    "⏳ To-Do": ("Planning",),
    "🚧 In-Progress": ("Active",),
    "🧱 Blocked": ("Blocked",),
    "💤 On-Hold": ("On-Hold",),
    "✅ Done": ("Completed",),
    "📦 Archived": ("Archived",),
}

ARCHIVED_TAG_NAMES: tuple[str, ...] = ("📦 Archived", "Archived")

STATUS_CHOICES: List[app_commands.Choice[str]] = [
    app_commands.Choice(name=meta["label"], value=key)
    for key, meta in BOARD_STATUS_META.items()
]


PRIORITY_META: dict[str, dict[str, Any]] = {
    "p0": {"label": "🔥 Now", "emoji": "🔥"},
    "p1": {"label": "🟠 Next", "emoji": "🟠"},
    "p2": {"label": "🟢 Later", "emoji": "🟢"},
}

PRIORITY_CANONICAL: dict[str, str] = {
    key: meta["label"] for key, meta in PRIORITY_META.items()
}

PRIORITY_ALIASES: dict[str, str] = {}
for key, canonical in PRIORITY_CANONICAL.items():
    label_without_emoji = canonical.split(" ", 1)[-1]
    PRIORITY_ALIASES[key] = key
    PRIORITY_ALIASES[canonical.casefold()] = key
    PRIORITY_ALIASES[label_without_emoji.casefold()] = key
    PRIORITY_ALIASES[label_without_emoji.replace(":", "").casefold()] = key
    PRIORITY_ALIASES[label_without_emoji.replace(" ", "").casefold()] = key
    PRIORITY_ALIASES[canonical.replace(" ", "").casefold()] = key
    PRIORITY_ALIASES[canonical.replace(":", "").casefold()] = key
legacy_priority_aliases = {
    "high": "p0",
    "priority: high": "p0",
    "priority high": "p0",
    "priority-high": "p0",
    "medium": "p1",
    "priority: medium": "p1",
    "priority medium": "p1",
    "priority-medium": "p1",
    "low": "p2",
    "priority: low": "p2",
    "priority low": "p2",
    "priority-low": "p2",
    "p0": "p0",
    "p1": "p1",
    "p2": "p2",
    "priority:high": "p0",
    "priority:medium": "p1",
    "priority:low": "p2",
    "🔥 p0": "p0",
    "🟠 p1": "p1",
    "🟢 p2": "p2",
    "now": "p0",
    "next": "p1",
    "later": "p2",
}
PRIORITY_ALIASES.update({alias.casefold(): value for alias, value in legacy_priority_aliases.items()})
for alias in ("urgent", "critical"):
    PRIORITY_ALIASES[alias] = "p0"

PROJECT_TYPE_META: dict[str, dict[str, Any]] = {
    "commission": {"label": "💼 Commission", "emoji": "💼", "aliases": ["commission", "client", "paid"]},
    "personal": {"label": "🎨 Personal", "emoji": "🎨", "aliases": ["personal", "solo"]},
    "collaboration": {"label": "🤝 Collaboration", "emoji": "🤝", "aliases": ["collaboration", "collab", "partner"]},
    "community": {
        "label": "🌱 Community",
        "emoji": "🌱",
        "aliases": ["community", "open source", "open-source", "oss"],
    },
    "internal": {"label": "🏢 Internal", "emoji": "🏢", "aliases": ["internal", "operations", "ops", "team"]},
    "study": {"label": "🧪 Study", "emoji": "🧪", "aliases": ["study", "research", "learning"]},
    "holiday": {
        "label": HOLIDAY_TAG_NAME,
        "emoji": "🎁",
        "aliases": ["holiday", "seasonal"],
    },
}

PROJECT_TYPE_CANONICAL: dict[str, str] = {
    key: meta["label"] for key, meta in PROJECT_TYPE_META.items()
}

PROJECT_TYPE_ALIASES: dict[str, str] = {}
for key, meta in PROJECT_TYPE_META.items():
    label = meta["label"]
    without_emoji = label.split(" ", 1)[-1]
    PROJECT_TYPE_ALIASES[key] = key
    PROJECT_TYPE_ALIASES[label.casefold()] = key
    PROJECT_TYPE_ALIASES[label.replace(" ", "").casefold()] = key
    PROJECT_TYPE_ALIASES[without_emoji.casefold()] = key
    PROJECT_TYPE_ALIASES[without_emoji.replace(" ", "").casefold()] = key
    for alias in meta.get("aliases", []):
        PROJECT_TYPE_ALIASES[alias.casefold()] = key

for meta in PRIORITY_META.values():
    REQUIRED_TAGS.setdefault(meta["label"], {"emoji": meta.get("emoji")})
for meta in PROJECT_TYPE_META.values():
    REQUIRED_TAGS.setdefault(meta["label"], {"emoji": meta.get("emoji")})

PRIORITY_CHOICES: List[app_commands.Choice[str]] = [
    app_commands.Choice(name=meta["label"], value=key)
    for key, meta in PRIORITY_META.items()
]
PRIORITY_CHOICES.append(app_commands.Choice(name="Clear priority tag", value="clear"))

PROJECT_TYPE_CHOICES: List[app_commands.Choice[str]] = [
    app_commands.Choice(name=meta["label"], value=key)
    for key, meta in PROJECT_TYPE_META.items()
]
PROJECT_TYPE_CHOICES.append(
    app_commands.Choice(name="Clear project type tag", value="clear")
)

_MISSING = object()


@dataclass(slots=True)
class ProjectRecord:
    """Representation of a project stored in SQLite."""

    project_id: int
    guild_id: int | None
    thread_id: int | None
    name: str
    summary: str | None
    owner_id: int | None
    status: str
    due_date: datetime | None
    holiday: bool
    tags: list[str]
    priority: str | None
    project_type: str | None
    scheduled_event_id: int | None
    created_at: datetime
    updated_at: datetime
    archived_at: datetime | None

    def as_display_lines(self) -> list[str]:
        """Return a list of human readable lines for embeds."""

        due = self.due_date.isoformat() if self.due_date else "No due date"
        status_key = self.status.lower()
        status_meta = BOARD_STATUS_META.get(status_key)
        status_label = (
            status_meta["label"]
            if status_meta
            else self.status.replace("-", " ").title()
        )
        owner = f"<@{self.owner_id}>" if self.owner_id else "Unassigned"
        tags = ", ".join(self.tags) if self.tags else "None"
        state = (
            BOARD_STATUS_META["archived"]["label"]
            if self.archived_at
            else status_label
        )
        return [
            f"**{self.name}**",
            f"• Status: {state}",
            f"• Owner: {owner}",
            f"• Due: {due}",
            f"• Tags: {tags}",
        ]


class ProjectsBoardView(discord.ui.View):
    """Persistent view used to control the projects board."""

    def __init__(
        self,
        board: "ProjectsBoard",
        *,
        board_id: int,
        records: list[ProjectRecord],
        selected_project_id: int | None,
        holiday_only: bool,
    ) -> None:
        super().__init__(timeout=None)
        self.board = board
        self.board_id = board_id
        self.records = records
        self.selected_project_id = selected_project_id
        self.holiday_only = holiday_only

        self.add_item(self.ProjectSelect(self))
        self.add_item(self.ActionSelect(self))
        self.add_item(self.RefreshButton(self))
        self.add_item(self.HolidayButton(self))
        self.add_item(self.ClearSelectionButton(self))

    # ------------------------------------------------------------------
    # Component helpers
    # ------------------------------------------------------------------
    def _project_options(self) -> list[discord.SelectOption]:
        options: list[discord.SelectOption] = []
        for record in self.records:
            due_label = self.board._format_due_label(record.due_date)
            status_label = self.board._status_display_name(record.status)
            description = f"{status_label} · {due_label}"
            options.append(
                discord.SelectOption(
                    label=f"#{record.project_id} · {record.name}",
                    value=str(record.project_id),
                    description=description[:100],
                    default=self.selected_project_id == record.project_id,
                )
            )
        if not options:
            options.append(
                discord.SelectOption(
                    label="No projects available",
                    value="noop",
                    description="Create a project to get started",
                    default=False,
                )
            )
        return options

    def _action_options(self) -> list[discord.SelectOption]:
        record = next(
            (proj for proj in self.records if proj.project_id == self.selected_project_id),
            None,
        )
        if record is None:
            return [
                discord.SelectOption(
                    label="Select a project first",
                    value="noop",
                    description="Choose a project to perform actions",
                    default=True,
                )
            ]

        current_status = self.board._normalise_status_key(record.status)

        options: list[discord.SelectOption] = []
        for status_key in BOARD_STATUS_ORDER:
            label = self.board._status_display_name(status_key)
            options.append(
                discord.SelectOption(
                    label=f"Set status → {label}",
                    value=f"status:{status_key}",
                    default=current_status == status_key,
                )
            )

        priority = self.board._priority_for_record(record)
        for key, meta in PRIORITY_META.items():
            options.append(
                discord.SelectOption(
                    label=f"Priority → {meta['label']}",
                    value=f"priority:{key}",
                    emoji=meta.get("emoji"),
                    default=priority == key,
                )
            )
        options.append(
            discord.SelectOption(
                label="Clear priority",
                value="priority:clear",
                emoji="✖️",
                default=priority is None,
            )
        )

        project_type = self.board._project_type_for_record(record)
        for key, meta in PROJECT_TYPE_META.items():
            options.append(
                discord.SelectOption(
                    label=f"Type → {meta['label']}",
                    value=f"type:{key}",
                    emoji=meta.get("emoji"),
                    default=project_type == key,
                )
            )
        options.append(
            discord.SelectOption(
                label="Clear project type",
                value="type:clear",
                emoji="✖️",
                default=project_type is None,
            )
        )

        options.append(
            discord.SelectOption(
                label="Archive project",
                value="archive",
                emoji="📦",
            )
        )
        return options

    # ------------------------------------------------------------------
    # Interaction helpers
    # ------------------------------------------------------------------
    async def handle_project_selected(self, interaction: discord.Interaction, project_id: int) -> None:
        await self.board._handle_project_selection(interaction, self.board_id, project_id)

    async def handle_action_selected(self, interaction: discord.Interaction, value: str) -> None:
        await self.board._handle_action_selection(
            interaction, self.board_id, self.selected_project_id, value
        )

    async def handle_refresh(self, interaction: discord.Interaction) -> None:
        await self.board._handle_refresh(interaction, self.board_id)

    async def handle_toggle_holiday(self, interaction: discord.Interaction) -> None:
        await self.board._handle_toggle_holiday(interaction, self.board_id)

    async def handle_clear_selection(self, interaction: discord.Interaction) -> None:
        await self.board._handle_clear_selection(interaction, self.board_id)

    # ------------------------------------------------------------------
    # Component definitions
    # ------------------------------------------------------------------
    class ProjectSelect(discord.ui.Select):
        def __init__(self, view: "ProjectsBoardView") -> None:
            options = view._project_options()
            disabled = len(view.records) == 0
            super().__init__(
                placeholder="Select a project",
                min_values=1,
                max_values=1,
                options=options,
                custom_id=f"proj_select:{view.board_id}",
                disabled=disabled,
            )
            self.view: ProjectsBoardView = view

        async def callback(self, interaction: discord.Interaction) -> None:
            if not self.values or self.values[0] == "noop":
                await interaction.response.send_message(
                    "No selectable projects are available.", ephemeral=True
                )
                return
            project_id = int(self.values[0])
            await self.view.handle_project_selected(interaction, project_id)

    class ActionSelect(discord.ui.Select):
        def __init__(self, view: "ProjectsBoardView") -> None:
            options = view._action_options()
            disabled = view.selected_project_id is None
            custom_id = (
                f"proj_action:{view.selected_project_id}" if view.selected_project_id else "proj_action:0"
            )
            super().__init__(
                placeholder="Choose an action",
                min_values=1,
                max_values=1,
                options=options,
                custom_id=custom_id,
                disabled=disabled,
                row=1,
            )
            self.view: ProjectsBoardView = view

        async def callback(self, interaction: discord.Interaction) -> None:
            if not self.values:
                return
            value = self.values[0]
            if value == "noop":
                await interaction.response.send_message(
                    "Select a project before choosing an action.", ephemeral=True
                )
                return
            await self.view.handle_action_selected(interaction, value)

    class RefreshButton(discord.ui.Button):
        def __init__(self, view: "ProjectsBoardView") -> None:
            super().__init__(
                style=discord.ButtonStyle.primary,
                label="Refresh",
                custom_id="btn_refresh",
                row=2,
            )
            self.view: ProjectsBoardView = view

        async def callback(self, interaction: discord.Interaction) -> None:
            await self.view.handle_refresh(interaction)

    class HolidayButton(discord.ui.Button):
        def __init__(self, view: "ProjectsBoardView") -> None:
            label = "Holiday Filter: On" if view.holiday_only else "Holiday Filter: Off"
            style = (
                discord.ButtonStyle.success if view.holiday_only else discord.ButtonStyle.secondary
            )
            super().__init__(label=label, style=style, custom_id="btn_holiday", row=2)
            self.view: ProjectsBoardView = view

        async def callback(self, interaction: discord.Interaction) -> None:
            await self.view.handle_toggle_holiday(interaction)

    class ClearSelectionButton(discord.ui.Button):
        def __init__(self, view: "ProjectsBoardView") -> None:
            super().__init__(
                style=discord.ButtonStyle.secondary,
                label="Clear Selection",
                custom_id="btn_clear",
                row=2,
            )
            self.view: ProjectsBoardView = view

        async def callback(self, interaction: discord.Interaction) -> None:
            await self.view.handle_clear_selection(interaction)


class ProjectsBoard(commands.Cog):
    """Cog implementing a projects board backed by a forum channel."""

    project = app_commands.Group(
        name="project",
        description="Manage collaborative projects",
        guild_only=True,
    )

    def __init__(
        self,
        bot: commands.Bot,
        db_manager: DBManager | None = None,
        scheduler: GoalScheduler | None = None,
        *,
        forum_channel_id: int | None = None,
        index_channel_id: int | None = None,
        monitor_channel_id: int | None = None,
        require_events: bool = False,
    ) -> None:
        self.bot = bot
        self._db_manager = db_manager
        self._scheduler = scheduler or GoalScheduler(db_manager)
        self._db_path = get_settings().social_graph_db
        self._conn: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()
        self._forum_channel_id = (
            forum_channel_id
            if forum_channel_id is not None
            else self._resolve_forum_channel_id()
        )
        self._index_channel_id = index_channel_id
        self._monitor_channel_id = monitor_channel_id
        self._require_events = require_events
        self._startup_task: asyncio.Task[None] | None = None
        self._ready = asyncio.Event()
        self._board_filters: dict[int, bool] = {}
        self._board_selection: dict[int, int | None] = {}
        if self.bot.loop.is_running():
            self._startup_task = self.bot.loop.create_task(self._startup())
        else:  # pragma: no cover - only triggered in sync setup paths
            self._startup_task = asyncio.create_task(self._startup())

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    async def cog_load(self) -> None:
        """Register slash command group when the cog loads."""

        self.bot.tree.add_command(self.project)
        if self._startup_task is None:
            self._startup_task = asyncio.create_task(self._startup())

    async def cog_unload(self) -> None:
        """Clean up resources when unloading."""

        self.bot.tree.remove_command(self.project.name, type=self.project.type)
        if self._startup_task:
            self._startup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._startup_task
        await self._close_db()

    async def _startup(self) -> None:
        await self._init_db()
        await self.bot.wait_until_ready()
        await self._sync_board_state()
        self._ready.set()

    async def _init_db(self) -> None:
        if self._conn is not None:
            return
        self._conn = await aiosqlite.connect(self._db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._ensure_schema()
        await self._conn.commit()

    async def _ensure_schema(self) -> None:
        if self._conn is None:
            raise RuntimeError("Database connection not initialized")
        await self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS projects (
                project_id INTEGER PRIMARY KEY AUTOINCREMENT,
                guild_id INTEGER,
                thread_id INTEGER UNIQUE,
                name TEXT NOT NULL,
                summary TEXT,
                owner_id INTEGER,
                status TEXT NOT NULL,
                due_date TEXT,
                holiday INTEGER DEFAULT 0,
                tags TEXT,
                priority TEXT,
                project_type TEXT,
                scheduled_event_id INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                archived_at TEXT
            )
            """
        )
        await self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_projects_status ON projects(status)
            """
        )
        await self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_projects_guild ON projects(guild_id)
            """
        )
        columns = await self._get_table_columns("projects")
        alterations: list[str] = []
        if "guild_id" not in columns:
            alterations.append("ALTER TABLE projects ADD COLUMN guild_id INTEGER")
        if "priority" not in columns:
            alterations.append("ALTER TABLE projects ADD COLUMN priority TEXT")
        if "project_type" not in columns:
            alterations.append("ALTER TABLE projects ADD COLUMN project_type TEXT")
        for statement in alterations:
            await self._conn.execute(statement)
        await self._backfill_project_metadata()

    async def _get_table_columns(self, table: str) -> set[str]:
        if self._conn is None:
            raise RuntimeError("Database connection not initialized")
        cursor = await self._conn.execute(f"PRAGMA table_info({table})")
        rows = await cursor.fetchall()
        await cursor.close()
        return {row["name"] for row in rows}

    async def _backfill_project_metadata(self, default_guild_id: int | None = None) -> None:
        if self._conn is None:
            raise RuntimeError("Database connection not initialized")
        cursor = await self._conn.execute(
            """
            SELECT project_id, guild_id, priority, project_type, tags
            FROM projects
            """
        )
        rows = await cursor.fetchall()
        await cursor.close()
        if not rows:
            return
        guild_id = default_guild_id or self._infer_default_guild_id()
        updates: list[tuple[Any, ...]] = []
        for row in rows:
            tags = json.loads(row["tags"]) if row["tags"] else []
            priority = self._priority_from_tags(tags)
            project_type = self._project_type_from_tags(tags)
            desired_guild_id = row["guild_id"] if row["guild_id"] else guild_id
            needs_update = False
            current_priority = row["priority"]
            current_project_type = row["project_type"]
            current_guild_id = row["guild_id"]
            if priority != current_priority:
                needs_update = True
            if project_type != current_project_type:
                needs_update = True
            if desired_guild_id is not None and desired_guild_id != current_guild_id:
                needs_update = True
            if not needs_update:
                continue
            updates.append((priority, project_type, desired_guild_id, row["project_id"]))
        if not updates:
            return
        await self._conn.executemany(
            """
            UPDATE projects
            SET priority = ?, project_type = ?, guild_id = COALESCE(?, guild_id)
            WHERE project_id = ?
            """,
            updates,
        )
        await self._conn.commit()

    def _infer_default_guild_id(self) -> int | None:
        channel_ids = [self._forum_channel_id, self._index_channel_id, self._monitor_channel_id]
        for channel_id in channel_ids:
            if not channel_id:
                continue
            channel = self.bot.get_channel(channel_id)
            if channel and getattr(channel, "guild", None):
                return channel.guild.id
        if getattr(self.bot, "guilds", None):
            guilds = getattr(self.bot, "guilds")
            if isinstance(guilds, list) and len(guilds) == 1 and getattr(guilds[0], "id", None):
                return guilds[0].id
        return None

    async def _close_db(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------
    def _resolve_forum_channel_id(self) -> int | None:
        settings = get_settings()
        setting_value = getattr(settings, "projects_forum_channel_id", None)
        if setting_value:
            return int(setting_value)
        for env_name in (
            "PROJECT_FORUM_CHANNEL_ID",
            "DT_PROJECT_FORUM_CHANNEL_ID",
            "PROJECTS_FORUM_CHANNEL_ID",
            "DT_PROJECTS_FORUM_CHANNEL_ID",
            "PROJECTS_FORUM_CHANNEL",
            "DT_PROJECTS_FORUM_CHANNEL",
        ):
            value = os.getenv(env_name)
            if value:
                with contextlib.suppress(ValueError):
                    return int(value)
        return None

    async def _fetch_forum_channel(self) -> discord.ForumChannel | None:
        if self._forum_channel_id is None:
            return None
        channel = self.bot.get_channel(self._forum_channel_id)
        if channel is None:
            with contextlib.suppress(discord.HTTPException, discord.NotFound, discord.Forbidden):
                channel = await self.bot.fetch_channel(self._forum_channel_id)
        if not isinstance(channel, discord.ForumChannel):
            _LOG.warning("Configured projects channel %s is not a forum", channel)
            return None
        return channel

    async def _sync_board_state(self) -> None:
        """Ensure tags and index embed are up to date."""

        channel = await self._fetch_forum_channel()
        if channel is None:
            return
        await self._backfill_project_metadata(channel.guild.id)
        created = await self._ensure_tags(channel)
        if created:
            _LOG.info("Created %d missing project tags", len(created))
        await self._update_index_embed(channel)

    async def _ensure_tags(self, channel: discord.ForumChannel) -> list[str]:
        existing = {tag.name: tag for tag in channel.available_tags}
        created: list[str] = []
        for name, meta in REQUIRED_TAGS.items():
            if name in existing:
                continue
            emoji = meta.get("emoji")
            try:
                tag = await channel.create_tag(name=name, emoji=emoji)
            except discord.Forbidden:
                _LOG.warning("Missing permissions to create forum tag %s", name)
                continue
            except discord.HTTPException as exc:
                _LOG.warning("Failed to create forum tag %s: %s", name, exc)
                continue
            existing[name] = tag
            created.append(name)
        return created

    # ------------------------------------------------------------------
    # Database helpers
    # ------------------------------------------------------------------
    async def _execute(self, query: str, *params: Any) -> aiosqlite.Cursor:
        if self._conn is None:
            raise RuntimeError("Database connection not initialized")
        return await self._conn.execute(query, params)

    async def _commit(self) -> None:
        if self._conn is not None:
            await self._conn.commit()

    async def _fetch_projects(
        self, *, include_archived: bool = False, guild_id: int | None = None
    ) -> list[ProjectRecord]:
        query = "SELECT * FROM projects"
        clauses: list[str] = []
        params: list[Any] = []
        if guild_id is not None:
            clauses.append("(guild_id = ? OR guild_id IS NULL)")
            params.append(guild_id)
        if not include_archived:
            clauses.append("archived_at IS NULL")
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += (
            " ORDER BY CASE priority"
            " WHEN 'p0' THEN 0"
            " WHEN 'p1' THEN 1"
            " WHEN 'p2' THEN 2"
            " ELSE 3 END, COALESCE(due_date, created_at) ASC, project_id ASC"
        )
        cursor = await self._execute(query, *params)
        rows = await cursor.fetchall()
        await cursor.close()
        return [self._row_to_project(row) for row in rows]

    async def _fetch_project(self, project_id: int) -> ProjectRecord | None:
        cursor = await self._execute("SELECT * FROM projects WHERE project_id = ?", project_id)
        row = await cursor.fetchone()
        await cursor.close()
        return self._row_to_project(row) if row else None

    async def _fetch_project_by_thread(self, thread_id: int) -> ProjectRecord | None:
        cursor = await self._execute("SELECT * FROM projects WHERE thread_id = ?", thread_id)
        row = await cursor.fetchone()
        await cursor.close()
        return self._row_to_project(row) if row else None

    def _row_to_project(self, row: aiosqlite.Row) -> ProjectRecord:
        due = self._parse_datetime(row["due_date"])
        created = self._parse_datetime(row["created_at"], required=True)
        updated = self._parse_datetime(row["updated_at"], required=True)
        archived = self._parse_datetime(row["archived_at"])
        tags = json.loads(row["tags"]) if row["tags"] else []
        return ProjectRecord(
            project_id=row["project_id"],
            guild_id=row["guild_id"],
            thread_id=row["thread_id"],
            name=row["name"],
            summary=row["summary"],
            owner_id=row["owner_id"],
            status=row["status"],
            due_date=due,
            holiday=bool(row["holiday"]),
            tags=list(tags),
            priority=row["priority"],
            project_type=row["project_type"],
            scheduled_event_id=row["scheduled_event_id"],
            created_at=created,
            updated_at=updated,
            archived_at=archived,
        )

    def _parse_datetime(self, value: Any, *, required: bool = False) -> datetime | None:
        if value is None:
            if required:
                raise ValueError("Expected timestamp")
            return None
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=UTC)
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=UTC)
        text = str(value)
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed

    def _serialize_datetime(self, value: datetime | None) -> str | None:
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")

    # ------------------------------------------------------------------
    # Slash command handlers
    # ------------------------------------------------------------------
    @project.command(name="seed_tags", description="Create the required forum tags if they are missing")
    async def seed_tags(self, interaction: discord.Interaction) -> None:
        await self._ready.wait()
        channel = await self._fetch_forum_channel()
        if channel is None:
            await interaction.response.send_message(
                "Projects forum channel is not configured.", ephemeral=True
            )
            return
        created = await self._ensure_tags(channel)
        if created:
            message = f"Created tags: {', '.join(created)}"
        else:
            message = "All required tags already exist."
        await interaction.response.send_message(message, ephemeral=True)

    @project.command(name="list", description="List active projects")
    @app_commands.describe(include_archived="Include archived projects in the listing")
    async def list_projects(self, interaction: discord.Interaction, include_archived: bool = False) -> None:
        await self._ready.wait()
        records = await self._fetch_projects(
            include_archived=include_archived, guild_id=interaction.guild_id
        )
        embed = discord.Embed(title="Projects", colour=discord.Colour.blurple())
        if not records:
            embed.description = "No projects recorded yet."
        else:
            for proj in records:
                embed.add_field(
                    name=f"#{proj.project_id} · {proj.name}",
                    value="\n".join(proj.as_display_lines()),
                    inline=False,
                )
        await interaction.response.send_message(embed=embed, ephemeral=True)

    @project.command(name="create", description="Create a new project and forum thread")
    @app_commands.describe(
        name="Title of the project",
        summary="Short project summary",
        status="Optional status for the project",
        due_date="Due date in ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM)",
        owner="Assign the project to a member",
        tags="Comma separated additional forum tags",
        holiday="Mark the project as holiday related",
        priority="Set the priority tag for the project",
        project_type="Assign a project type tag",
    )
    @app_commands.choices(
        status=STATUS_CHOICES,
        priority=PRIORITY_CHOICES,
        project_type=PROJECT_TYPE_CHOICES,
    )
    async def create_project(
        self,
        interaction: discord.Interaction,
        name: str,
        summary: str | None = None,
        status: app_commands.Choice[str] | None = None,
        due_date: str | None = None,
        owner: discord.Member | None = None,
        tags: str | None = None,
        holiday: bool = False,
        priority: app_commands.Choice[str] | None = None,
        project_type: app_commands.Choice[str] | None = None,
    ) -> None:
        await self._ready.wait()
        channel = await self._fetch_forum_channel()
        if channel is None:
            await interaction.response.send_message(
                "Projects forum channel is not configured.", ephemeral=True
            )
            return
        parsed_due = self._parse_due_date_input(due_date)
        if due_date and parsed_due is None:
            await interaction.response.send_message(
                "Unable to parse due date. Use YYYY-MM-DD or ISO 8601 format.",
                ephemeral=True,
            )
            return
        status_value = status.value if status else DEFAULT_STATUS
        priority_value = self._resolve_priority_choice(priority)
        project_type_value = self._resolve_project_type_choice(project_type)
        async with self._lock:
            record = await self._create_project_record(
                channel=channel,
                name=name,
                summary=summary,
                status=status_value,
                due_date=parsed_due,
                owner=owner,
                tags=tags,
                holiday=holiday,
                priority=priority_value,
                project_type=project_type_value,
            )
        await interaction.response.send_message(
            f"Created project #{record.project_id}: {record.name}", ephemeral=True
        )
        await self._update_index_embed(channel)
        await self._sync_due_date_reminder(record, channel.guild if channel else None)

    @project.command(name="update", description="Update an existing project")
    @app_commands.describe(
        project_id="Project identifier from /project list",
        name="New project name",
        summary="Updated summary",
        status="Updated status",
        due_date="New due date (empty string to clear)",
        owner="Reassign the project",
        tags="Comma separated tag names (replaces existing ones)",
        holiday="Mark project as holiday themed",
        priority="Update the project's priority tag",
        project_type="Update the project type tag",
    )
    @app_commands.choices(
        status=STATUS_CHOICES,
        priority=PRIORITY_CHOICES,
        project_type=PROJECT_TYPE_CHOICES,
    )
    async def update_project(
        self,
        interaction: discord.Interaction,
        project_id: int,
        name: str | None = None,
        summary: str | None = None,
        status: app_commands.Choice[str] | None = None,
        due_date: str | None = None,
        owner: discord.Member | None = None,
        tags: str | None = None,
        holiday: bool | None = None,
        priority: app_commands.Choice[str] | None = None,
        project_type: app_commands.Choice[str] | None = None,
    ) -> None:
        await self._ready.wait()
        channel = await self._fetch_forum_channel()
        if channel is None:
            await interaction.response.send_message(
                "Projects forum channel is not configured.", ephemeral=True
            )
            return
        parsed_due = self._parse_due_date_input(due_date)
        if due_date and parsed_due is None and due_date.strip():
            await interaction.response.send_message(
                "Unable to parse due date. Use YYYY-MM-DD or ISO 8601 format.",
                ephemeral=True,
            )
            return
        priority_value = self._resolve_priority_choice(priority)
        project_type_value = self._resolve_project_type_choice(project_type)
        async with self._lock:
            record = await self._update_project_record(
                project_id,
                channel=channel,
                name=name,
                summary=summary,
                status=status.value if status else None,
                due_date=parsed_due if due_date else None,
                owner=owner,
                tags=tags,
                holiday=holiday,
                clear_due=due_date == "",
                priority=priority_value,
                project_type=project_type_value,
            )
        if record is None:
            await interaction.response.send_message(
                "Project not found.", ephemeral=True
            )
            return
        await interaction.response.send_message(
            f"Updated project #{record.project_id}.", ephemeral=True
        )
        await self._update_index_embed(channel)
        await self._sync_due_date_reminder(record, channel.guild if channel else None)

    @project.command(name="archive", description="Archive a project and mark its thread")
    @app_commands.describe(project_id="Identifier of the project to archive")
    async def archive_project(self, interaction: discord.Interaction, project_id: int) -> None:
        await self._ready.wait()
        channel = await self._fetch_forum_channel()
        if channel is None:
            await interaction.response.send_message(
                "Projects forum channel is not configured.", ephemeral=True
            )
            return
        async with self._lock:
            record = await self._archive_project(project_id, channel)
        if record is None:
            await interaction.response.send_message("Project not found.", ephemeral=True)
            return
        await interaction.response.send_message(
            f"Archived project #{record.project_id}.", ephemeral=True
        )
        await self._update_index_embed(channel)
        await self._sync_due_date_reminder(record, channel.guild if channel else None)

    # ------------------------------------------------------------------
    # Project mutations
    # ------------------------------------------------------------------
    async def _create_project_record(
        self,
        *,
        channel: discord.ForumChannel,
        name: str,
        summary: str | None,
        status: str,
        due_date: datetime | None,
        owner: discord.Member | None,
        tags: str | None,
        holiday: bool,
        priority: str | None | object = _MISSING,
        project_type: str | None | object = _MISSING,
    ) -> ProjectRecord:
        applied_tags = await self._resolve_tags(
            channel,
            status,
            tags,
            holiday,
            due_date,
            priority=priority,
            project_type=project_type,
        )
        content = summary or "Project created via /project create"
        thread = None
        try:
            thread = await channel.create_thread(
                name=name,
                content=content,
                applied_tags=applied_tags,
            )
        except discord.Forbidden:
            _LOG.warning("Missing permissions to create project thread")
        except discord.HTTPException as exc:
            _LOG.warning("Failed to create project thread: %s", exc)
        tag_names = [tag.name for tag in applied_tags]
        due_serialized = self._serialize_datetime(due_date)
        owner_id = owner.id if owner else None
        now = datetime.now(UTC)
        guild_id = getattr(channel.guild, "id", None)
        canonical_priority = (
            priority if priority is not _MISSING else self._priority_from_tags(tag_names)
        )
        canonical_project_type = (
            project_type
            if project_type is not _MISSING
            else self._project_type_from_tags(tag_names)
        )
        cursor = await self._execute(
            """
            INSERT INTO projects (
                guild_id,
                thread_id,
                name,
                summary,
                owner_id,
                status,
                due_date,
                holiday,
                tags,
                priority,
                project_type,
                scheduled_event_id,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)
            """,
            guild_id,
            thread.id if thread else None,
            name,
            summary,
            owner_id,
            status,
            due_serialized,
            int(holiday or self._is_holiday_project(due_date, tag_names)),
            json.dumps(tag_names) if tag_names else None,
            canonical_priority,
            canonical_project_type,
            self._serialize_datetime(now),
            self._serialize_datetime(now),
        )
        project_id = cursor.lastrowid
        await cursor.close()
        await self._commit()
        record = await self._fetch_project(project_id)
        if thread and owner_id:
            with contextlib.suppress(discord.HTTPException, discord.Forbidden):
                await thread.add_user(owner)
        return record

    async def _update_project_record(
        self,
        project_id: int,
        *,
        channel: discord.ForumChannel,
        name: str | None,
        summary: str | None,
        status: str | None,
        due_date: datetime | None,
        owner: discord.Member | None,
        tags: str | None,
        holiday: bool | None,
        clear_due: bool,
        priority: str | None | object = _MISSING,
        project_type: str | None | object = _MISSING,
    ) -> ProjectRecord | None:
        record = await self._fetch_project(project_id)
        if record is None:
            return None
        thread = await self._fetch_thread(channel, record.thread_id)
        new_status = status or record.status
        new_name = name or record.name
        new_summary = summary if summary is not None else record.summary
        new_due = due_date if due_date is not None else (None if clear_due else record.due_date)
        new_owner_id = owner.id if owner is not None else record.owner_id
        new_holiday = holiday if holiday is not None else record.holiday
        applied_tags = await self._resolve_tags(
            channel,
            new_status,
            tags,
            new_holiday,
            new_due,
            priority=priority,
            project_type=project_type,
            existing=record.tags,
        )
        tag_names = [tag.name for tag in applied_tags]
        if thread:
            await self._update_thread(thread, new_name, new_summary, applied_tags, owner)
        now = datetime.now(UTC)
        due_serialized = self._serialize_datetime(new_due)
        resolved_guild_id = record.guild_id or getattr(channel.guild, "id", None)
        canonical_priority = self._priority_from_tags(tag_names)
        canonical_project_type = self._project_type_from_tags(tag_names)
        await self._execute(
            """
            UPDATE projects
            SET guild_id = COALESCE(?, guild_id), name = ?, summary = ?, owner_id = ?, status = ?, due_date = ?, holiday = ?, tags = ?, priority = ?, project_type = ?, updated_at = ?
            WHERE project_id = ?
            """,
            resolved_guild_id,
            new_name,
            new_summary,
            new_owner_id,
            new_status,
            due_serialized,
            int(new_holiday or self._is_holiday_project(new_due, tag_names)),
            json.dumps(tag_names) if tag_names else None,
            canonical_priority,
            canonical_project_type,
            self._serialize_datetime(now),
            project_id,
        )
        await self._commit()
        updated = await self._fetch_project(project_id)
        return updated

    async def _archive_project(
        self, project_id: int, channel: discord.ForumChannel
    ) -> ProjectRecord | None:
        record = await self._fetch_project(project_id)
        if record is None:
            return None
        thread = await self._fetch_thread(channel, record.thread_id)
        now = datetime.now(UTC)
        archived_at = self._serialize_datetime(now)
        applied_tags = await self._resolve_tags(
            channel,
            "archived",
            None,
            record.holiday,
            record.due_date,
            existing=record.tags,
        )
        await self._execute(
            """
            UPDATE projects
            SET status = 'archived', archived_at = ?, tags = ?, updated_at = ?
            WHERE project_id = ?
            """,
            archived_at,
            json.dumps([tag.name for tag in applied_tags]) if applied_tags else None,
            archived_at,
            project_id,
        )
        await self._commit()
        if thread:
            with contextlib.suppress(discord.HTTPException, discord.Forbidden):
                await thread.edit(archived=True, locked=True, applied_tags=applied_tags)
        updated = await self._fetch_project(project_id)
        if updated and updated.scheduled_event_id:
            await self._cancel_scheduled_event(updated, channel.guild if channel else None)
        return updated

    async def _set_project_priority(
        self, project_id: int, priority: str | None, channel: discord.ForumChannel
    ) -> ProjectRecord | None:
        record = await self._fetch_project(project_id)
        if record is None:
            return None
        tag_names = self._apply_priority_tags(record.tags, priority)
        applied_tags = await self._resolve_tags(
            channel,
            record.status,
            None,
            record.holiday,
            record.due_date,
            priority=priority,
            existing=tag_names,
        )
        now = datetime.now(UTC)
        new_tag_names = [tag.name for tag in applied_tags]
        canonical_priority = self._priority_from_tags(new_tag_names)
        await self._execute(
            """
            UPDATE projects
            SET tags = ?, priority = ?, updated_at = ?
            WHERE project_id = ?
            """,
            json.dumps(new_tag_names) if applied_tags else None,
            canonical_priority,
            self._serialize_datetime(now),
            project_id,
        )
        await self._commit()
        thread = await self._fetch_thread(channel, record.thread_id)
        if thread:
            with contextlib.suppress(discord.HTTPException, discord.Forbidden):
                await thread.edit(applied_tags=applied_tags)
        return await self._fetch_project(project_id)

    async def _set_project_type(
        self, project_id: int, project_type: str | None, channel: discord.ForumChannel
    ) -> ProjectRecord | None:
        record = await self._fetch_project(project_id)
        if record is None:
            return None
        tag_names = self._apply_project_type_tags(record.tags, project_type)
        applied_tags = await self._resolve_tags(
            channel,
            record.status,
            None,
            record.holiday,
            record.due_date,
            project_type=project_type,
            existing=tag_names,
        )
        now = datetime.now(UTC)
        new_tag_names = [tag.name for tag in applied_tags]
        canonical_project_type = self._project_type_from_tags(new_tag_names)
        await self._execute(
            """
            UPDATE projects
            SET tags = ?, project_type = ?, updated_at = ?
            WHERE project_id = ?
            """,
            json.dumps(new_tag_names) if applied_tags else None,
            canonical_project_type,
            self._serialize_datetime(now),
            project_id,
        )
        await self._commit()
        thread = await self._fetch_thread(channel, record.thread_id)
        if thread:
            with contextlib.suppress(discord.HTTPException, discord.Forbidden):
                await thread.edit(applied_tags=applied_tags)
        return await self._fetch_project(project_id)

    async def _fetch_thread(
        self, channel: discord.ForumChannel, thread_id: int | None
    ) -> discord.Thread | None:
        if thread_id is None:
            return None
        thread = channel.get_thread(thread_id)
        if thread is None:
            with contextlib.suppress(discord.HTTPException, discord.NotFound, discord.Forbidden):
                thread = await channel.fetch_thread(thread_id)
        return thread

    async def _update_thread(
        self,
        thread: discord.Thread,
        name: str,
        summary: str | None,
        applied_tags: list[discord.ForumTag],
        owner: discord.Member | None,
    ) -> None:
        with contextlib.suppress(discord.HTTPException, discord.Forbidden):
            await thread.edit(name=name, applied_tags=applied_tags)
        if summary:
            try:
                message = await thread.fetch_message(thread.id)
            except discord.NotFound:
                message = None
            except discord.HTTPException:
                message = None
            if message is not None:
                with contextlib.suppress(discord.HTTPException, discord.Forbidden):
                    await message.edit(content=summary)
        if owner is not None:
            with contextlib.suppress(discord.HTTPException, discord.Forbidden):
                await thread.add_user(owner)

    async def _resolve_tags(
        self,
        channel: discord.ForumChannel,
        status: str,
        tags: str | None,
        holiday: bool,
        due_date: datetime | None,
        *,
        priority: str | None | object = _MISSING,
        project_type: str | None | object = _MISSING,
        existing: Iterable[str] | None = None,
    ) -> list[discord.ForumTag]:
        await self._ensure_tags(channel)
        available = {tag.name: tag for tag in channel.available_tags}
        initial_tags = list(existing or [])
        normalised = self._apply_priority_tags(initial_tags, priority)
        normalised = self._apply_project_type_tags(normalised, project_type)
        tag_names = set(normalised)
        status_tag = STATUS_TAGS.get(status.lower(), STATUS_TAGS[DEFAULT_STATUS])
        status_candidates = (status_tag,) + LEGACY_STATUS_TAG_NAMES.get(status_tag, ())
        for candidate in status_candidates:
            if candidate in available:
                for archived_name in ARCHIVED_TAG_NAMES:
                    tag_names.discard(archived_name)
                tag_names.add(candidate)
                break
        additional = self._parse_tags(tags)
        for extra in additional:
            if extra in available:
                tag_names.add(extra)
        if holiday or self._is_holiday_project(due_date, tag_names):
            for candidate in (HOLIDAY_TAG_NAME, "Holiday"):
                if candidate in available:
                    tag_names.add(candidate)
                    break
        resolved = [available[name] for name in tag_names if name in available]
        return resolved

    def _parse_tags(self, tags: str | None) -> list[str]:
        if not tags:
            return []
        values = [tag.strip() for tag in tags.split(",")]
        return [tag for tag in values if tag]

    def _resolve_priority_choice(
        self, choice: app_commands.Choice[str] | None
    ) -> str | None | object:
        if choice is None:
            return _MISSING
        value = choice.value
        if value == "clear":
            return None
        mapped = PRIORITY_ALIASES.get(str(value).casefold())
        if mapped:
            return mapped
        if value in PRIORITY_CANONICAL:
            return value
        return _MISSING

    def _resolve_project_type_choice(
        self, choice: app_commands.Choice[str] | None
    ) -> str | None | object:
        if choice is None:
            return _MISSING
        value = choice.value
        if value == "clear":
            return None
        mapped = PROJECT_TYPE_ALIASES.get(str(value).casefold())
        if mapped:
            return mapped
        if value in PROJECT_TYPE_CANONICAL:
            return value
        return _MISSING

    def _is_holiday_project(
        self, due_date: datetime | None, tag_names: Iterable[str]
    ) -> bool:
        if due_date and (
            _is_halloween_window(due_date)
            or _is_thanksgiving_window(due_date)
            or _is_christmas_new_year_window(due_date)
            or _is_valentines_window(due_date)
        ):
            return True
        normalized = {tag.strip().casefold() for tag in tag_names}
        return any(alias in normalized for alias in HOLIDAY_TAG_ALIASES)

    # ------------------------------------------------------------------
    # Index embed management
    # ------------------------------------------------------------------
    async def _update_index_embed(
        self,
        channel: discord.ForumChannel,
        *,
        interaction: discord.Interaction | None = None,
        selected_project_id: int | None | object = _MISSING,
        holiday_only: bool | None = None,
    ) -> None:
        records = await self._fetch_projects(
            include_archived=False, guild_id=getattr(channel.guild, "id", None)
        )
        thread, message = await self._ensure_index_message(channel)
        if thread is None and interaction is None:
            return
        message_id: int | None = None
        if interaction and interaction.message:
            message_id = interaction.message.id
        elif message is not None:
            message_id = message.id
        elif thread is not None:
            message_id = thread.id

        if message_id is None:
            return

        current_filter = self._board_filters.get(message_id, False)
        holiday_flag = current_filter if holiday_only is None else holiday_only
        self._board_filters[message_id] = holiday_flag

        filtered_records = (
            [record for record in records if record.holiday]
            if holiday_flag
            else list(records)
        )

        valid_ids = {record.project_id for record in filtered_records}
        if selected_project_id is _MISSING:
            selection = self._board_selection.get(message_id)
        else:
            selection = selected_project_id
        if selection not in valid_ids:
            selection = None
        self._board_selection[message_id] = selection

        embed = self._build_board_embed(filtered_records, holiday_only=holiday_flag)
        view = ProjectsBoardView(
            self,
            board_id=message_id,
            records=filtered_records,
            selected_project_id=selection,
            holiday_only=holiday_flag,
        )

        if interaction is not None:
            if interaction.response.is_done():
                await interaction.edit_original_response(embed=embed, view=view)
            else:
                await interaction.response.edit_message(embed=embed, view=view)
            updated_message_id = interaction.message.id if interaction.message else None
        else:
            updated_message_id = None
            if message is None and thread is not None:
                try:
                    message = await thread.send(embed=embed, view=view)
                except (discord.HTTPException, discord.Forbidden):
                    message = None
            elif message is not None:
                with contextlib.suppress(discord.HTTPException, discord.Forbidden):
                    await message.edit(embed=embed, view=view)
            if message is not None:
                updated_message_id = message.id

        final_message_id = updated_message_id or message_id
        if final_message_id is not None:
            self.bot.add_view(view, message_id=final_message_id)

    def _build_board_embed(
        self, records: list[ProjectRecord], *, holiday_only: bool
    ) -> discord.Embed:
        description = "Pinned overview of active projects"
        if holiday_only:
            description = "Holiday projects view — showing only holiday-tagged efforts"
        embed = discord.Embed(
            title="Projects Board Index",
            description=description,
            colour=discord.Colour.teal(),
        )

        def to_utc(value: datetime | None) -> datetime | None:
            if value is None:
                return None
            if value.tzinfo is None:
                return value.replace(tzinfo=UTC)
            return value.astimezone(UTC)

        now = datetime.now(UTC)
        soon_cutoff = now + timedelta(days=7)
        recent_cutoff = now - timedelta(days=30)
        far_future = datetime.max.replace(tzinfo=UTC)

        now_bucket: list[ProjectRecord] = []
        next_bucket: list[ProjectRecord] = []
        holiday_bucket: list[ProjectRecord] = []
        done_bucket: list[ProjectRecord] = []

        high_priority_keys = {"p0"}
        legacy_high_key = PRIORITY_ALIASES.get("high")
        if legacy_high_key:
            high_priority_keys.add(legacy_high_key)

        for record in records:
            status_key = self._normalise_status_key(record.status)
            priority = self._priority_for_record(record)
            due_date = to_utc(record.due_date)
            is_holiday = record.holiday or self._is_holiday_project(due_date, record.tags)
            if is_holiday:
                holiday_bucket.append(record)

            if status_key == "done":
                done_bucket.append(record)
                continue

            if (
                (priority and priority in high_priority_keys)
                or (due_date is not None and due_date <= soon_cutoff)
            ):
                now_bucket.append(record)
            else:
                next_bucket.append(record)

        def due_sort(record: ProjectRecord) -> tuple[int, datetime, int]:
            due = to_utc(record.due_date)
            return (0 if due else 1, due or far_future, record.project_id)

        now_bucket.sort(key=due_sort)
        next_bucket.sort(key=due_sort)

        displayed_ids = {record.project_id for record in now_bucket + next_bucket}
        holiday_unique = [
            record for record in sorted(holiday_bucket, key=due_sort) if record.project_id not in displayed_ids
        ]

        done_bucket.sort(
            key=lambda record: to_utc(record.updated_at) or datetime.min.replace(tzinfo=UTC),
            reverse=True,
        )
        recent_done = [
            record
            for record in done_bucket
            if (to_utc(record.updated_at) or datetime.min.replace(tzinfo=UTC)) >= recent_cutoff
        ]
        if recent_done:
            done_bucket = recent_done[:5]
        else:
            done_bucket = done_bucket[:5]

        def format_section(values: Iterable[ProjectRecord], empty: str) -> str:
            lines = [self._format_index_entry(record) for record in values]
            if not lines:
                return empty
            return "\n".join(lines)

        if not records:
            embed.description = (
                "No holiday projects right now."
                if holiday_only
                else "No active projects yet. Use /project create to add one."
            )

        embed.add_field(
            name="🔥 Now",
            value=format_section(now_bucket, "_Nothing marked as high priority right now._"),
            inline=False,
        )
        embed.add_field(
            name="🟠 Next",
            value=format_section(next_bucket, "_No upcoming projects in the queue._"),
            inline=False,
        )
        holiday_empty_message = "_No holiday projects on the radar._"
        if holiday_only or (holiday_bucket and not holiday_unique):
            holiday_empty_message = "_All holiday projects are listed above._"
        embed.add_field(
            name="🎁 Holiday Radar",
            value=format_section(holiday_unique, holiday_empty_message),
            inline=False,
        )
        embed.add_field(
            name="✅ Recently Done",
            value=format_section(done_bucket, "_Nothing completed recently._"),
            inline=False,
        )

        if holiday_only:
            embed.set_footer(text="Holiday filter enabled")
        return embed

    # ------------------------------------------------------------------
    # Board interaction handlers
    # ------------------------------------------------------------------
    async def _handle_project_selection(
        self, interaction: discord.Interaction, board_id: int, project_id: int
    ) -> None:
        await self._ready.wait()
        channel = await self._fetch_forum_channel()
        if channel is None:
            await self._respond_ephemeral(
                interaction, "Projects forum channel is not configured."
            )
            return
        await self._update_index_embed(
            channel,
            interaction=interaction,
            selected_project_id=project_id,
        )

    async def _handle_action_selection(
        self,
        interaction: discord.Interaction,
        board_id: int,
        project_id: int | None,
        value: str,
    ) -> None:
        await self._ready.wait()
        if project_id is None:
            await self._respond_ephemeral(
                interaction, "Select a project before choosing an action."
            )
            return
        channel = await self._fetch_forum_channel()
        if channel is None:
            await self._respond_ephemeral(
                interaction, "Projects forum channel is not configured."
            )
            return

        updated: ProjectRecord | None = None
        response_message: str | None = None
        new_selection: int | None = project_id

        if value.startswith("status:"):
            status_value = value.split(":", 1)[1]
            async with self._lock:
                updated = await self._update_project_record(
                    project_id,
                    channel=channel,
                    name=None,
                    summary=None,
                    status=status_value,
                    due_date=None,
                    owner=None,
                    tags=None,
                    holiday=None,
                    clear_due=False,
                )
            if updated:
                status_label = self._status_display_name(status_value)
                response_message = f"Set status to {status_label}."
        elif value.startswith("priority:"):
            priority_value = value.split(":", 1)[1]
            if priority_value == "clear":
                priority_choice: str | None = None
            elif priority_value in PRIORITY_CANONICAL:
                priority_choice = priority_value
            else:
                await self._respond_ephemeral(interaction, "Unknown priority selection.")
                return
            async with self._lock:
                updated = await self._set_project_priority(project_id, priority_choice, channel)
            if updated:
                response_message = (
                    "Cleared priority."
                    if priority_choice is None
                    else f"Set priority to {PRIORITY_CANONICAL[priority_choice]}"
                )
        elif value.startswith("type:"):
            type_value = value.split(":", 1)[1]
            if type_value == "clear":
                type_choice: str | None = None
            elif type_value in PROJECT_TYPE_CANONICAL:
                type_choice = type_value
            else:
                await self._respond_ephemeral(interaction, "Unknown project type selection.")
                return
            async with self._lock:
                updated = await self._set_project_type(project_id, type_choice, channel)
            if updated:
                response_message = (
                    "Cleared project type."
                    if type_choice is None
                    else f"Set project type to {PROJECT_TYPE_CANONICAL[type_choice]}"
                )
        elif value == "archive":
            async with self._lock:
                updated = await self._archive_project(project_id, channel)
            new_selection = None
            if updated:
                response_message = "Archived project."
        else:
            await self._respond_ephemeral(interaction, "Unsupported action selected.")
            return

        if updated is None:
            await self._respond_ephemeral(interaction, "Project not found or could not be updated.")
            return

        await self._sync_due_date_reminder(updated, channel.guild if channel else None)
        await self._update_index_embed(
            channel,
            interaction=interaction,
            selected_project_id=new_selection,
        )
        if response_message:
            await interaction.followup.send(response_message, ephemeral=True)

    async def _handle_refresh(
        self, interaction: discord.Interaction, board_id: int
    ) -> None:
        await self._ready.wait()
        channel = await self._fetch_forum_channel()
        if channel is None:
            await self._respond_ephemeral(
                interaction, "Projects forum channel is not configured."
            )
            return
        await self._update_index_embed(channel, interaction=interaction)

    async def _handle_toggle_holiday(
        self, interaction: discord.Interaction, board_id: int
    ) -> None:
        await self._ready.wait()
        channel = await self._fetch_forum_channel()
        if channel is None:
            await self._respond_ephemeral(
                interaction, "Projects forum channel is not configured."
            )
            return
        message_id = interaction.message.id if interaction.message else board_id
        current = self._board_filters.get(message_id, False)
        await self._update_index_embed(
            channel,
            interaction=interaction,
            holiday_only=not current,
        )

    async def _handle_clear_selection(
        self, interaction: discord.Interaction, board_id: int
    ) -> None:
        await self._ready.wait()
        channel = await self._fetch_forum_channel()
        if channel is None:
            await self._respond_ephemeral(
                interaction, "Projects forum channel is not configured."
            )
            return
        await self._update_index_embed(
            channel,
            interaction=interaction,
            selected_project_id=None,
        )

    def _format_status_bucket(self, records: Iterable[ProjectRecord]) -> str:
        lines = [self._format_index_entry(record) for record in records]
        if not lines:
            return "_No projects in this column._"
        return "\n".join(lines)

    def _format_index_entry(self, record: ProjectRecord) -> str:
        indicators: list[str] = []
        priority_indicator = self._priority_indicator_for_record(record)
        if priority_indicator:
            indicators.append(priority_indicator)
        type_indicator = self._project_type_indicator_for_record(record)
        if type_indicator:
            indicators.append(type_indicator)
        if record.holiday:
            indicators.append("🎄")
        indicator_text = f"{' '.join(indicators)} " if indicators else ""
        due_label = self._format_due_label(record.due_date)
        return f"• {indicator_text}[#{record.project_id}] {record.name} ({due_label})"

    def _priority_for_record(self, record: ProjectRecord) -> str | None:
        return record.priority or self._priority_from_tags(record.tags)

    def _priority_indicator_for_record(self, record: ProjectRecord) -> str:
        priority_key = self._priority_for_record(record)
        if not priority_key:
            return ""
        meta = PRIORITY_META.get(priority_key)
        return meta.get("emoji", "") if meta else ""

    def _priority_indicator_from_tags(self, tags: Iterable[str]) -> str:
        priority_key = self._priority_from_tags(tags)
        if not priority_key:
            return ""
        meta = PRIORITY_META.get(priority_key)
        return meta.get("emoji", "") if meta else ""

    def _priority_from_tags(self, tags: Iterable[str]) -> str | None:
        for tag in tags:
            key = PRIORITY_ALIASES.get(tag.strip().casefold())
            if key:
                return key
        return None

    def _apply_priority_tags(
        self, tags: Iterable[str], priority: str | None | object
    ) -> list[str]:
        cleaned: list[str] = []
        for tag in tags:
            if PRIORITY_ALIASES.get(tag.strip().casefold()):
                continue
            cleaned.append(tag)
        current = self._priority_from_tags(tags)
        if priority is _MISSING:
            desired = current
        else:
            desired = priority
        if desired and isinstance(desired, str):
            canonical = PRIORITY_CANONICAL.get(desired)
            if canonical and canonical not in cleaned:
                cleaned.append(canonical)
        return cleaned

    def _project_type_for_record(self, record: ProjectRecord) -> str | None:
        return record.project_type or self._project_type_from_tags(record.tags)

    def _project_type_from_tags(self, tags: Iterable[str]) -> str | None:
        for tag in tags:
            key = PROJECT_TYPE_ALIASES.get(tag.strip().casefold())
            if key:
                return key
        return None

    def _apply_project_type_tags(
        self, tags: Iterable[str], project_type: str | None | object
    ) -> list[str]:
        cleaned: list[str] = []
        for tag in tags:
            if PROJECT_TYPE_ALIASES.get(tag.strip().casefold()):
                continue
            cleaned.append(tag)
        current = self._project_type_from_tags(tags)
        if project_type is _MISSING:
            desired = current
        else:
            desired = project_type
        if desired and isinstance(desired, str):
            canonical = PROJECT_TYPE_CANONICAL.get(desired)
            if canonical and canonical not in cleaned:
                cleaned.append(canonical)
        return cleaned

    def _project_type_indicator_for_record(self, record: ProjectRecord) -> str:
        type_key = self._project_type_for_record(record)
        if not type_key:
            return ""
        meta = PROJECT_TYPE_META.get(type_key)
        return meta.get("emoji", "") if meta else ""

    def _project_type_indicator_from_tags(self, tags: Iterable[str]) -> str:
        type_key = self._project_type_from_tags(tags)
        if not type_key:
            return ""
        meta = PROJECT_TYPE_META.get(type_key)
        return meta.get("emoji", "") if meta else ""

    def _format_due_label(self, due_date: datetime | None) -> str:
        if due_date is None:
            return "no due date"
        if due_date.tzinfo is None:
            due_date = due_date.replace(tzinfo=UTC)
        due_date = due_date.astimezone(UTC)
        formatted = due_date.strftime("%b %d").replace(" 0", " ")
        current_year = datetime.now(UTC).year
        if due_date.year != current_year:
            formatted += f" {due_date.year}"
        return f"due {formatted}"

    def _normalise_status_key(self, status: str) -> str:
        if not status:
            return "to-do"
        normalised = status.strip().lower()
        normalised = normalised.replace(" ", "-").replace("_", "-")
        aliases = {
            "todo": "to-do",
            "to-do": "to-do",
            "backlog": "to-do",
            "planning": "to-do",
            "inprogress": "in-progress",
            "in-progress": "in-progress",
            "active": "in-progress",
            "ongoing": "in-progress",
            "onhold": "on-hold",
            "on-hold": "on-hold",
            "completed": "done",
            "complete": "done",
            "finished": "done",
        }
        return aliases.get(normalised, normalised)

    def _status_display_name(self, status_key: str) -> str:
        canonical = self._normalise_status_key(status_key)
        meta = BOARD_STATUS_META.get(canonical)
        if meta:
            return meta["label"]
        return canonical.replace("-", " ").title()

    async def _locate_index_thread(
        self, channel: discord.ForumChannel
    ) -> tuple[discord.Thread | None, discord.TextChannel | None]:
        if self._index_channel_id is not None:
            index_channel = self.bot.get_channel(self._index_channel_id)
            if index_channel is None:
                with contextlib.suppress(discord.HTTPException, discord.NotFound, discord.Forbidden):
                    index_channel = await self.bot.fetch_channel(self._index_channel_id)
            if isinstance(index_channel, discord.Thread):
                return index_channel, None
            if isinstance(index_channel, discord.TextChannel):
                return None, index_channel
            if index_channel is not None:
                _LOG.warning(
                    "Configured projects index channel %s is not a supported channel type",
                    index_channel,
                )
        for thread in channel.threads:
            if thread.name == INDEX_THREAD_NAME:
                return thread, None
        # Attempt to load archived threads where the index might live
        async for thread in channel.archived_threads(limit=50, private=False):
            if thread.name == INDEX_THREAD_NAME:
                return thread, None
        return None, None

    async def _ensure_index_message(
        self, channel: discord.ForumChannel
    ) -> tuple[discord.Thread | None, discord.Message | None]:
        thread, message_channel = await self._locate_index_thread(channel)
        if message_channel is not None:
            message = await self._fetch_index_message(message_channel)
            if message is None:
                with contextlib.suppress(discord.HTTPException, discord.Forbidden):
                    message = await message_channel.send(content="Projects board index")
            if message is not None and not message.pinned:
                with contextlib.suppress(discord.HTTPException, discord.Forbidden):
                    await message.pin()
            if message is not None:
                return None, message
            _LOG.warning(
                "Failed to locate or create index message in configured channel %s; falling back to forum thread",
                message_channel,
            )

        if thread is None:
            thread = await self._create_index_thread(channel)
        if thread is None:
            return None, None
        message = await self._fetch_index_message(thread)
        if message is None:
            with contextlib.suppress(discord.HTTPException, discord.Forbidden):
                message = await thread.send(content="Projects board index")
        if thread.archived or not thread.pinned:
            with contextlib.suppress(discord.HTTPException, discord.Forbidden):
                await thread.edit(pinned=True, archived=False)
        return thread, message

    async def _fetch_index_message(
        self, target: discord.Thread | discord.TextChannel
    ) -> discord.Message | None:
        if isinstance(target, discord.Thread):
            try:
                return await target.fetch_message(target.id)
            except (discord.NotFound, discord.HTTPException):
                return None

        pins: list[discord.Message] = []
        with contextlib.suppress(discord.HTTPException, discord.Forbidden):
            pins = await target.pins()
        for message in pins:
            if self._is_index_message(message):
                return message
        try:
            async for message in target.history(limit=25):
                if self._is_index_message(message):
                    return message
        except (discord.Forbidden, discord.HTTPException):
            return None
        return None

    def _is_index_message(self, message: discord.Message) -> bool:
        bot_user = self.bot.user
        if bot_user is not None and message.author.id != bot_user.id:
            return False
        if message.content and message.content.strip().casefold() == "projects board index":
            return True
        for embed in message.embeds:
            title = embed.title or ""
            if title.strip().casefold() == "projects board index":
                return True
        return False

    async def _create_index_thread(
        self, channel: discord.ForumChannel
    ) -> discord.Thread | None:
        try:
            thread = await channel.create_thread(
                name=INDEX_THREAD_NAME,
                content="Projects board index",
                applied_tags=[],
            )
        except discord.Forbidden:
            _LOG.warning("Missing permissions to create index thread")
            return None
        except discord.HTTPException as exc:
            _LOG.warning("Failed to create index thread: %s", exc)
            return None
        with contextlib.suppress(discord.HTTPException, discord.Forbidden):
            await thread.edit(pinned=True, archived=False)
        return thread

    # ------------------------------------------------------------------
    # Due date reminders
    # ------------------------------------------------------------------
    async def _sync_due_date_reminder(
        self, record: ProjectRecord, guild: discord.Guild | None
    ) -> None:
        if record is None:
            return
        if record.archived_at or record.due_date is None:
            if guild is not None:
                await self._cancel_scheduled_event(record, guild)
            return

        due_date = _normalize_datetime(record.due_date)
        reminder_time = due_date - DEFAULT_REMINDER_LEAD
        now = datetime.now(UTC)
        if reminder_time < now:
            reminder_time = now + timedelta(minutes=5)

        if not self._require_events or guild is None:
            if guild is not None:
                await self._cancel_scheduled_event(record, guild)
            self._queue_goal_scheduler_reminder(record, reminder_time)
            return

        await self._create_or_update_event(record, guild, reminder_time)

    async def _create_or_update_event(
        self, record: ProjectRecord, guild: discord.Guild, start_time: datetime
    ) -> None:
        description = record.summary or "Project due date reminder"
        end_time = record.due_date
        if end_time is None:
            return
        start_time = _normalize_datetime(start_time)
        end_time = _normalize_datetime(end_time)
        now = datetime.now(UTC)

        if end_time <= now:
            await self._cancel_scheduled_event(record, guild)
            self._queue_goal_scheduler_reminder(record, now)
            return

        if start_time >= end_time:
            adjusted_start = end_time - DEFAULT_REMINDER_LEAD
            if adjusted_start <= now:
                adjusted_start = now + timedelta(minutes=5)
            if adjusted_start >= end_time:
                await self._cancel_scheduled_event(record, guild)
                self._queue_goal_scheduler_reminder(record, now)
                return
            start_time = adjusted_start
        try:
            if record.scheduled_event_id:
                event = guild.get_scheduled_event(record.scheduled_event_id)
                if event is None:
                    event = await guild.fetch_scheduled_event(record.scheduled_event_id)
                await event.edit(
                    name=f"{record.name} deadline",
                    description=description,
                    start_time=start_time,
                    end_time=end_time,
                    entity_type=discord.EntityType.external,
                    location=SCHEDULED_EVENT_LOCATION,
                )
            else:
                event = await guild.create_scheduled_event(
                    name=f"{record.name} deadline",
                    start_time=start_time,
                    end_time=end_time,
                    description=description,
                    privacy_level=discord.PrivacyLevel.guild_only,
                    entity_type=discord.EntityType.external,
                    location=SCHEDULED_EVENT_LOCATION,
                )
                await self._set_scheduled_event_id(record.project_id, event.id)
        except discord.Forbidden:
            _LOG.warning("Missing permissions to manage scheduled events for %s", record.name)
            self._queue_goal_scheduler_reminder(record, start_time)
        except discord.HTTPException as exc:
            _LOG.warning("Failed to manage scheduled event for %s: %s", record.name, exc)
            self._queue_goal_scheduler_reminder(record, start_time)

    def _queue_goal_scheduler_reminder(
        self, record: ProjectRecord, reminder_time: datetime
    ) -> None:
        due_display = (
            _normalize_datetime(record.due_date).isoformat()
            if record.due_date is not None
            else "unspecified"
        )
        reminder_at = _normalize_datetime(reminder_time)
        now = datetime.now(UTC)
        delay_seconds = max(0, (reminder_at - now).total_seconds())
        delay = int(delay_seconds)
        if delay == 0:
            reminder_at = now
        reminder_text = reminder_at.isoformat()
        message = (
            f"Reminder: project {record.name} (ID #{record.project_id}) is due {due_display} "
            f"(schedule at {reminder_text}, {DEFAULT_REMINDER_LEAD} ahead)."
        delay_seconds = max(0, math.ceil((reminder_at - now).total_seconds()))
        reminder_message = (
            f"Reminder: project {record.name} is due {due_display} "
            f"(schedule at {reminder_text}, {DEFAULT_REMINDER_LEAD} ahead)"
        )
        self._scheduler.add_goal(
            f"{delay_seconds}:{reminder_message}",
            priority=5,
        )
        if record.thread_id:
            message = f"{message} Discuss in <#{record.thread_id}>."
        else:
            message = (
                f"{message} Follow up via the projects board for project #{record.project_id}."
            )
        self._scheduler.add_goal(f"{delay}:{message}", priority=5)

    async def _cancel_scheduled_event(
        self, record: ProjectRecord, guild: discord.Guild | None
    ) -> None:
        if guild is None or not record.scheduled_event_id:
            return
        try:
            event = guild.get_scheduled_event(record.scheduled_event_id)
            if event is None:
                event = await guild.fetch_scheduled_event(record.scheduled_event_id)
            if event is not None:
                await event.cancel()
        except discord.HTTPException as exc:
            _LOG.debug("Failed to cancel scheduled event %s: %s", record.scheduled_event_id, exc)
        finally:
            await self._set_scheduled_event_id(record.project_id, None)

    async def _set_scheduled_event_id(
        self, project_id: int, event_id: int | None
    ) -> None:
        await self._execute(
            "UPDATE projects SET scheduled_event_id = ? WHERE project_id = ?",
            event_id,
            project_id,
        )
        await self._commit()

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _parse_due_date_input(self, value: str | None) -> datetime | None:
        if not value:
            return None
        text = value.strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y"):
                with contextlib.suppress(ValueError):
                    parsed = datetime.strptime(text, fmt)
                    break
            else:
                return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed

    async def _respond_ephemeral(
        self, interaction: discord.Interaction, message: str
    ) -> None:
        if interaction.response.is_done():
            await interaction.followup.send(message, ephemeral=True)
        else:
            await interaction.response.send_message(message, ephemeral=True)


async def setup(bot: commands.Bot) -> None:
    """Register the :class:`ProjectsBoard` cog with ``bot``."""

    await bot.add_cog(ProjectsBoard(bot))
