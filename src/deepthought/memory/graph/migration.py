from __future__ import annotations

import sqlite3
from typing import Any

from ..fact_extractor import extract_typed_fact_triples_from_turn
from .pipeline import ingest_conversation_turns
from .store import GraphMemoryStore


def migrate_sqlite_memories_to_graph(sqlite_path: str, store: GraphMemoryStore) -> dict[str, Any]:
    """Move rows from SQLite ``memories`` into graph facts with provenance tags."""

    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT user_id, topic, memory, sentiment_score, timestamp FROM memories ORDER BY timestamp ASC"
    ).fetchall()
    conn.close()

    turns = []
    for row in rows:
        turns.append(
            {
                "user_id": str(row["user_id"] or "anonymous"),
                "text": str(row["memory"] or ""),
                "timestamp": str(row["timestamp"] or ""),
                "input_id": f"sqlite-memory:{row['user_id']}:{row['timestamp']}",
            }
        )
        if row["topic"]:
            triples = extract_typed_fact_triples_from_turn(
                user_id=str(row["user_id"] or "anonymous"),
                message=f"My favorite topic is {row['topic']}",
                timestamp=str(row["timestamp"] or ""),
                source_id=f"sqlite-topic:{row['user_id']}:{row['timestamp']}",
            )
            for triple in triples:
                turns.append(
                    {
                        "user_id": triple["subject_id"],
                        "text": f"{triple['predicate']} {triple.get('object_value')}",
                        "timestamp": str(row["timestamp"] or ""),
                        "input_id": f"sqlite-derived:{triple['subject_id']}:{triple['predicate']}",
                    }
                )

    upserts = ingest_conversation_turns(turns, store)
    return {"rows_read": len(rows), "graph_upserts": upserts}
