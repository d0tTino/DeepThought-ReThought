"""Discord projects board management cog."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
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
DEFAULT_STATUS = "active"
DEFAULT_REMINDER_LEAD = timedelta(hours=1)
HOLIDAY_MONTHS = {11, 12}

REQUIRED_TAGS: dict[str, dict[str, str | None]] = {
    "Active": {"emoji": "🚧"},
    "Planning": {"emoji": "🧠"},
    "Blocked": {"emoji": "⛔"},
    "Completed": {"emoji": "✅"},
    "Archived": {"emoji": "🗃️"},
    "Holiday": {"emoji": "🎄"},
}

STATUS_TAGS: dict[str, str] = {
    "active": "Active",
    "planning": "Planning",
    "blocked": "Blocked",
    "completed": "Completed",
    "archived": "Archived",
}

STATUS_CHOICES: List[app_commands.Choice[str]] = [
    app_commands.Choice(name=name, value=value)
    for value, name in [
        ("active", "Active"),
        ("planning", "Planning"),
        ("blocked", "Blocked"),
        ("completed", "Completed"),
        ("archived", "Archived"),
    ]
]


@dataclass(slots=True)
class ProjectRecord:
    """Representation of a project stored in SQLite."""

    project_id: int
    thread_id: int | None
    name: str
    summary: str | None
    owner_id: int | None
    status: str
    due_date: datetime | None
    holiday: bool
    tags: list[str]
    scheduled_event_id: int | None
    created_at: datetime
    updated_at: datetime
    archived_at: datetime | None

    def as_display_lines(self) -> list[str]:
        """Return a list of human readable lines for embeds."""

        due = self.due_date.isoformat() if self.due_date else "No due date"
        status = self.status.capitalize()
        owner = f"<@{self.owner_id}>" if self.owner_id else "Unassigned"
        tags = ", ".join(self.tags) if self.tags else "None"
        state = "Archived" if self.archived_at else status
        return [
            f"**{self.name}**",
            f"• Status: {state}",
            f"• Owner: {owner}",
            f"• Due: {due}",
            f"• Tags: {tags}",
        ]


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
        await self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS projects (
                project_id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id INTEGER UNIQUE,
                name TEXT NOT NULL,
                summary TEXT,
                owner_id INTEGER,
                status TEXT NOT NULL,
                due_date TEXT,
                holiday INTEGER DEFAULT 0,
                tags TEXT,
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
        await self._conn.commit()

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
        for env_name in ("PROJECTS_FORUM_CHANNEL_ID", "DT_PROJECTS_FORUM_CHANNEL_ID"):
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

    async def _fetch_projects(self, *, include_archived: bool = False) -> list[ProjectRecord]:
        query = "SELECT * FROM projects"
        if not include_archived:
            query += " WHERE archived_at IS NULL"
        query += " ORDER BY COALESCE(due_date, created_at) ASC"
        cursor = await self._execute(query)
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
            thread_id=row["thread_id"],
            name=row["name"],
            summary=row["summary"],
            owner_id=row["owner_id"],
            status=row["status"],
            due_date=due,
            holiday=bool(row["holiday"]),
            tags=list(tags),
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
        records = await self._fetch_projects(include_archived=include_archived)
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
    )
    @app_commands.choices(status=STATUS_CHOICES)
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
    )
    @app_commands.choices(status=STATUS_CHOICES)
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
    ) -> ProjectRecord:
        applied_tags = await self._resolve_tags(channel, status, tags, holiday, due_date)
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
        cursor = await self._execute(
            """
            INSERT INTO projects (thread_id, name, summary, owner_id, status, due_date, holiday, tags, scheduled_event_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)
            """,
            thread.id if thread else None,
            name,
            summary,
            owner_id,
            status,
            due_serialized,
            int(holiday or self._is_holiday_project(due_date, tag_names)),
            json.dumps(tag_names) if tag_names else None,
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
            existing=record.tags,
        )
        tag_names = [tag.name for tag in applied_tags]
        if thread:
            await self._update_thread(thread, new_name, new_summary, applied_tags, owner)
        now = datetime.now(UTC)
        due_serialized = self._serialize_datetime(new_due)
        await self._execute(
            """
            UPDATE projects
            SET name = ?, summary = ?, owner_id = ?, status = ?, due_date = ?, holiday = ?, tags = ?, updated_at = ?
            WHERE project_id = ?
            """,
            new_name,
            new_summary,
            new_owner_id,
            new_status,
            due_serialized,
            int(new_holiday or self._is_holiday_project(new_due, tag_names)),
            json.dumps(tag_names) if tag_names else None,
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
        existing: Iterable[str] | None = None,
    ) -> list[discord.ForumTag]:
        await self._ensure_tags(channel)
        available = {tag.name: tag for tag in channel.available_tags}
        tag_names = set(existing or [])
        status_tag = STATUS_TAGS.get(status.lower(), STATUS_TAGS[DEFAULT_STATUS])
        if status_tag in available:
            tag_names.discard("Archived")
            tag_names.add(status_tag)
        additional = self._parse_tags(tags)
        for extra in additional:
            if extra in available:
                tag_names.add(extra)
        if holiday or self._is_holiday_project(due_date, tag_names):
            if "Holiday" in available:
                tag_names.add("Holiday")
        resolved = [available[name] for name in tag_names if name in available]
        return resolved

    def _parse_tags(self, tags: str | None) -> list[str]:
        if not tags:
            return []
        values = [tag.strip() for tag in tags.split(",")]
        return [tag for tag in values if tag]

    def _is_holiday_project(
        self, due_date: datetime | None, tag_names: Iterable[str]
    ) -> bool:
        if due_date and due_date.month in HOLIDAY_MONTHS:
            return True
        return any(tag.lower() == "holiday" for tag in tag_names)

    # ------------------------------------------------------------------
    # Index embed management
    # ------------------------------------------------------------------
    async def _update_index_embed(self, channel: discord.ForumChannel) -> None:
        records = await self._fetch_projects(include_archived=False)
        embed = discord.Embed(
            title="Projects Board Index",
            description="Pinned overview of active projects",
            colour=discord.Colour.teal(),
        )

        ordered_statuses = ["to-do", "in-progress", "blocked", "on-hold", "done"]
        buckets: dict[str, list[ProjectRecord]] = {status: [] for status in ordered_statuses}
        extra_statuses: dict[str, list[ProjectRecord]] = {}

        for record in records:
            status_key = self._normalise_status_key(record.status)
            if status_key in buckets:
                buckets[status_key].append(record)
            else:
                extra_statuses.setdefault(status_key, []).append(record)

        if not records:
            embed.description = "No active projects yet. Use /project create to add one."

        for status in ordered_statuses:
            display_name = self._status_display_name(status)
            embed.add_field(
                name=display_name,
                value=self._format_status_bucket(buckets[status]),
                inline=False,
            )

        for status_key in sorted(extra_statuses, key=lambda key: self._status_display_name(key).casefold()):
            display_name = self._status_display_name(status_key)
            embed.add_field(
                name=display_name,
                value=self._format_status_bucket(extra_statuses[status_key]),
                inline=False,
            )
        thread = await self._locate_index_thread(channel)
        if thread is None:
            await self._create_index_thread(channel, embed)
            return
        await self._edit_index_thread(thread, embed)

    def _format_status_bucket(self, records: Iterable[ProjectRecord]) -> str:
        lines = [self._format_index_entry(record) for record in records]
        if not lines:
            return "_No projects in this column._"
        return "\n".join(lines)

    def _format_index_entry(self, record: ProjectRecord) -> str:
        indicators: list[str] = []
        priority_indicator = self._priority_indicator_from_tags(record.tags)
        if priority_indicator:
            indicators.append(priority_indicator)
        if record.holiday:
            indicators.append("🎄")
        indicator_text = f"{' '.join(indicators)} " if indicators else ""
        due_label = self._format_due_label(record.due_date)
        return f"• {indicator_text}[#{record.project_id}] {record.name} ({due_label})"

    def _priority_indicator_from_tags(self, tags: Iterable[str]) -> str:
        priority_map = [
            ("urgent", "‼️"),
            ("critical", "‼️"),
            ("priority: high", "‼️"),
            ("high priority", "‼️"),
            ("priority-high", "‼️"),
            ("priority: medium", "🔸"),
            ("medium priority", "🔸"),
            ("priority-medium", "🔸"),
            ("priority: low", "🔻"),
            ("low priority", "🔻"),
            ("priority-low", "🔻"),
        ]
        for tag in tags:
            lowered = tag.strip().lower()
            for needle, indicator in priority_map:
                if needle in lowered:
                    return indicator
        return ""

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
        display_map = {
            "to-do": "To-Do",
            "in-progress": "In-Progress",
            "blocked": "Blocked",
            "on-hold": "On-Hold",
            "done": "Done",
        }
        if status_key in display_map:
            return display_map[status_key]
        return status_key.replace("-", " ").title()

    async def _locate_index_thread(
        self, channel: discord.ForumChannel
    ) -> discord.Thread | None:
        if self._index_channel_id is not None:
            thread = self.bot.get_channel(self._index_channel_id)
            if thread is None:
                with contextlib.suppress(discord.HTTPException, discord.NotFound, discord.Forbidden):
                    thread = await self.bot.fetch_channel(self._index_channel_id)
            if isinstance(thread, discord.Thread):
                return thread
            if thread is not None:
                _LOG.warning("Configured projects index channel %s is not a thread", thread)
        for thread in channel.threads:
            if thread.name == INDEX_THREAD_NAME:
                return thread
        # Attempt to load archived threads where the index might live
        async for thread in channel.archived_threads(limit=50, private=False):
            if thread.name == INDEX_THREAD_NAME:
                return thread
        return None

    async def _create_index_thread(
        self, channel: discord.ForumChannel, embed: discord.Embed
    ) -> None:
        try:
            thread = await channel.create_thread(
                name=INDEX_THREAD_NAME,
                content="Projects board index",
                embed=embed,
                applied_tags=[],
            )
        except discord.Forbidden:
            _LOG.warning("Missing permissions to create index thread")
            return
        except discord.HTTPException as exc:
            _LOG.warning("Failed to create index thread: %s", exc)
            return
        with contextlib.suppress(discord.HTTPException, discord.Forbidden):
            await thread.edit(pinned=True, archived=False)

    async def _edit_index_thread(
        self, thread: discord.Thread, embed: discord.Embed
    ) -> None:
        try:
            message = await thread.fetch_message(thread.id)
        except discord.NotFound:
            message = None
        except discord.HTTPException:
            message = None
        if message is None:
            with contextlib.suppress(discord.HTTPException, discord.Forbidden):
                await thread.send(embed=embed)
            return
        with contextlib.suppress(discord.HTTPException, discord.Forbidden):
            await message.edit(embed=embed)
        if not thread.pinned:
            with contextlib.suppress(discord.HTTPException, discord.Forbidden):
                await thread.edit(pinned=True)

    # ------------------------------------------------------------------
    # Due date reminders
    # ------------------------------------------------------------------
    async def _sync_due_date_reminder(
        self, record: ProjectRecord, guild: discord.Guild | None
    ) -> None:
        if guild is None or record is None:
            return
        if not self._require_events:
            return
        if record.archived_at:
            await self._cancel_scheduled_event(record, guild)
            return
        if record.due_date is None:
            await self._cancel_scheduled_event(record, guild)
            return
        start_time = record.due_date - DEFAULT_REMINDER_LEAD
        if start_time < datetime.now(UTC):
            start_time = datetime.now(UTC) + timedelta(minutes=5)
        await self._create_or_update_event(record, guild, start_time)

    async def _create_or_update_event(
        self, record: ProjectRecord, guild: discord.Guild, start_time: datetime
    ) -> None:
        description = record.summary or "Project due date reminder"
        end_time = record.due_date
        if end_time is None:
            return
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=UTC)
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=UTC)
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
                )
            else:
                event = await guild.create_scheduled_event(
                    name=f"{record.name} deadline",
                    start_time=start_time,
                    end_time=end_time,
                    description=description,
                    privacy_level=discord.PrivacyLevel.guild_only,
                )
                await self._set_scheduled_event_id(record.project_id, event.id)
        except discord.Forbidden:
            _LOG.warning("Missing permissions to manage scheduled events for %s", record.name)
            self._scheduler.add_goal(
                f"Reminder: project {record.name} is due {record.due_date.isoformat() if record.due_date else ''}",
                priority=5,
            )
        except discord.HTTPException as exc:
            _LOG.warning("Failed to manage scheduled event for %s: %s", record.name, exc)
            self._scheduler.add_goal(
                f"Reminder: project {record.name} is due {record.due_date.isoformat() if record.due_date else ''}",
                priority=5,
            )

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


async def setup(bot: commands.Bot) -> None:
    """Register the :class:`ProjectsBoard` cog with ``bot``."""

    await bot.add_cog(ProjectsBoard(bot))
