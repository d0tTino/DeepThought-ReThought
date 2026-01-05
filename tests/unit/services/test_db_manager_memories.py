import pytest

pytest.importorskip("aiosqlite")

from deepthought.services.db_manager import DBManager


@pytest.mark.asyncio
async def test_recall_user_returns_newest_first_with_limit(tmp_path):
    db = DBManager(str(tmp_path / "db.sqlite"))
    await db.connect()
    await db.init_db()

    assert db._db
    await db._db.executemany(
        "INSERT INTO memories (user_id, topic, memory, sentiment_score, timestamp) VALUES (?, ?, ?, ?, ?)",
        [
            ("user", "", "first", None, "2024-01-01T00:00:00Z"),
            ("user", "", "second", None, "2024-01-02T00:00:00Z"),
            ("user", "", "third", None, "2024-01-03T00:00:00Z"),
        ],
    )
    await db._db.commit()

    rows = await db.recall_user("user", limit=2)

    assert [row[1] for row in rows] == ["third", "second"]

    await db.close()
