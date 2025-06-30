import sqlite3
from typing import Iterable, List, Tuple


class OfflineSearch:
    """Very small wrapper around a SQLite FTS5 index."""

    def __init__(self, db_path: str) -> None:
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row

    @classmethod
    def create_index(
        cls, db_path: str, docs: Iterable[Tuple[str, str]]
    ) -> "OfflineSearch":
        """Create or reuse an FTS5 index at ``db_path``."""
        conn = sqlite3.connect(db_path)
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS documents USING FTS5(title, content)"
        )
        conn.executemany(
            "INSERT INTO documents(title, content) VALUES (?, ?)", list(docs)
        )
        conn.commit()
        conn.close()
        return cls(db_path)

    def search(self, query: str, limit: int = 3) -> List[str]:
        cur = self._conn.execute(
            "SELECT content FROM documents WHERE documents MATCH ? LIMIT ?",
            (query, limit),
        )
        rows = cur.fetchall()
        return [str(r["content"]) for r in rows]
