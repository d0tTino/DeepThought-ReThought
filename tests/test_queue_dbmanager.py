import asyncio
import json

import aiosqlite
import pytest

import examples.social_graph_bot as sg


@pytest.mark.asyncio
async def test_db_manager_list_pending_tasks_only_pending(tmp_path):
    db_file = tmp_path / "db.sqlite"
    sg.db_manager = sg.DBManager(str(db_file))
    await sg.db_manager.connect()
    await sg.db_manager.init_db()

    ctx = {"channel_id": 1}
    done_task = await sg.queue_deep_reflection("u1", ctx, "hello1")
    pending_task = await sg.queue_deep_reflection("u2", ctx, "hello2")
    await sg.db_manager.mark_task_done(done_task)

    rows = await sg.db_manager.list_pending_tasks()
    assert (pending_task, "u2", json.dumps(ctx), "hello2") in rows
    assert all(row[0] != done_task for row in rows)

    await sg.db_manager.close()


@pytest.mark.asyncio
async def test_db_manager_mark_task_done_updates_status(tmp_path):
    db_file = tmp_path / "db.sqlite"
    sg.db_manager = sg.DBManager(str(db_file))
    await sg.db_manager.connect()
    await sg.db_manager.init_db()

    task_id = await sg.queue_deep_reflection("u1", {"channel_id": 1}, "hello")
    await sg.db_manager.mark_task_done(task_id)

    async with aiosqlite.connect(str(db_file)) as db:
        async with db.execute(
            "SELECT status FROM queued_tasks WHERE task_id=?",
            (task_id,),
        ) as cur:
            row = await cur.fetchone()
    assert row[0] == "done"

    await sg.db_manager.close()
