import asyncio
import os

import pytest

pytest.importorskip("aiosqlite")
import aiosqlite

import examples.social_graph_bot as sg

pytest.importorskip("nats")
from deepthought.services import DBManager


@pytest.mark.asyncio
async def test_connect_creates_directory(tmp_path):
    db_file = tmp_path / "subdir" / "db.sqlite"
    manager = DBManager(str(db_file))
    await manager.connect()
    assert db_file.parent.is_dir()
    assert os.path.exists(db_file)
    await manager.close()
