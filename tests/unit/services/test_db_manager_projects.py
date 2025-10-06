"""Tests for DBManager project helpers."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

import pytest

pytest.importorskip("aiosqlite")

MODULE_PATH = Path(__file__).resolve().parents[3] / "src" / "deepthought" / "services" / "db_manager.py"
package_root = MODULE_PATH.parent.parent
deepthought_pkg = types.ModuleType("deepthought")
deepthought_pkg.__path__ = [str(package_root)]  # type: ignore[attr-defined]
deepthought_spec = importlib.machinery.ModuleSpec("deepthought", loader=None, is_package=True)
deepthought_spec.submodule_search_locations = [str(package_root)]
deepthought_pkg.__spec__ = deepthought_spec
services_pkg = types.ModuleType("deepthought.services")
services_pkg.__path__ = [str(MODULE_PATH.parent)]  # type: ignore[attr-defined]
services_spec = importlib.machinery.ModuleSpec("deepthought.services", loader=None, is_package=True)
services_spec.submodule_search_locations = [str(MODULE_PATH.parent)]
services_pkg.__spec__ = services_spec
sys.modules.setdefault("deepthought", deepthought_pkg)
sys.modules.setdefault("deepthought.services", services_pkg)

config_path = package_root / "config.py"
config_spec = importlib.util.spec_from_file_location("deepthought.config", config_path)
assert config_spec and config_spec.loader
config_module = importlib.util.module_from_spec(config_spec)
sys.modules.setdefault("deepthought.config", config_module)
sys.modules.setdefault("deepthought.services.config", config_module)
config_spec.loader.exec_module(config_module)

SPEC = importlib.util.spec_from_file_location(
    "deepthought.services.db_manager",
    MODULE_PATH,
    submodule_search_locations=[str(MODULE_PATH.parent)],
)
assert SPEC and SPEC.loader
_db_manager = importlib.util.module_from_spec(SPEC)
sys.modules["deepthought.services.db_manager"] = _db_manager
SPEC.loader.exec_module(_db_manager)

DBManager = _db_manager.DBManager


@pytest.mark.asyncio
async def test_create_and_get_project_records_embed_ready_dict() -> None:
    manager = DBManager(":memory:")
    await manager.init_db()
    try:
        due_date = datetime(2024, 5, 1, 12, 0, tzinfo=timezone.utc)
        record = await manager.create_project(
            thread_id=1234567890,
            title="Launch Sequence",
            priority=5,
            status="active",
            due_date=due_date,
            holiday=True,
        )

        assert record["thread_id"] == 1234567890
        assert record["title"] == "Launch Sequence"
        assert record["priority"] == 5
        assert record["status"] == "active"
        assert record["holiday"] is True
        assert record["due_date"] is not None and "2024-05-01" in record["due_date"]
        assert record["created_at"] is not None and "T" in record["created_at"]
        assert record["updated_at"] is not None and "T" in record["updated_at"]
        assert record["archived_at"] is None

        fetched = await manager.get_project_by_thread(1234567890)
        assert fetched == record
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_update_and_archive_project_flow() -> None:
    manager = DBManager(":memory:")
    await manager.init_db()
    try:
        created = await manager.create_project(
            thread_id=222,
            title="Test Project",
            priority=1,
            status="planning",
        )

        updated = await manager.update_project(
            222,
            status="completed",
            due_date=None,
            holiday=False,
            priority=10,
        )
        assert updated is not None
        assert updated["status"] == "completed"
        assert updated["due_date"] is None
        assert updated["holiday"] is False
        assert updated["priority"] == 10
        assert updated["updated_at"] >= created["updated_at"]

        archived = await manager.archive_project(222)
        assert archived is not None
        assert archived["archived_at"] is not None
        assert archived["status"] == "archived"

        active_projects = await manager.list_projects()
        assert all(project["archived_at"] is None for project in active_projects)

        all_projects = await manager.list_projects(include_archived=True)
        assert len(all_projects) == 1
        assert all_projects[0]["archived_at"] is not None
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_list_projects_orders_by_priority_and_filters_archived() -> None:
    manager = DBManager(":memory:")
    await manager.init_db()
    try:
        await manager.create_project(
            thread_id=1,
            title="Low",
            priority=1,
            status="active",
        )
        await manager.create_project(
            thread_id=2,
            title="High",
            priority=5,
            status="active",
        )
        await manager.archive_project(1)

        active = await manager.list_projects()
        assert len(active) == 1
        assert active[0]["thread_id"] == 2

        all_projects = await manager.list_projects(include_archived=True)
        assert [project["priority"] for project in all_projects] == [5, 1]
    finally:
        await manager.close()
