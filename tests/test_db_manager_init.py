import ast
import types
from pathlib import Path

import aiosqlite
import pytest

TABLES = [
    "interactions",
    "affinity",
    "memories",
    "theories",
    "queued_tasks",
    "sentiment_trends",
    "themes",
    "user_flags",
    "recent_topics",
]


def load_dbmanager():
    path = Path(__file__).resolve().parents[1] / "examples" / "social_graph_bot.py"
    source = path.read_text()
    tree = ast.parse(source)
    namespace = {
        "aiosqlite": aiosqlite,
        "os": __import__("os"),
        "DB_PATH": str(path),
    }
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "DBManager":
            exec(compile(ast.Module([node], []), filename=str(path), mode="exec"), namespace)
            return namespace["DBManager"]
    raise RuntimeError("DBManager not found")


@pytest.mark.asyncio
async def test_db_manager_init_creates_tables_once(tmp_path):
    DBManager = load_dbmanager()
    db_file = tmp_path / "db.sqlite"
    manager = DBManager(str(db_file))
    await manager.init_db()

    async with aiosqlite.connect(str(db_file)) as db:
        for table in TABLES:
            async with db.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            ) as cur:
                row = await cur.fetchone()
            assert row[0] == 1, f"{table} table should exist exactly once"

    await manager.close()
