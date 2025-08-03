"""Utility to migrate legacy JSON user graphs to the SQLite database."""

from __future__ import annotations

import argparse
import asyncio
from typing import Dict, Tuple

from deepthought.services.db_manager import DBManager
from deepthought.services.user_graph_dal import UserGraphDAL


async def migrate(json_path: str, db_path: str) -> None:
    dal = UserGraphDAL(json_path)
    db = DBManager(db_path)
    await db.connect()
    await db.init_db()

    # Migrate node affinities
    assert dal._graph  # for type checkers
    for node, data in dal._graph.nodes(data=True):
        affinity = int(data.get("affinity", 0))
        if affinity:
            await db._db.execute(
                """
                INSERT INTO affinity (user_id, score)
                VALUES (?, ?)
                ON CONFLICT(user_id) DO UPDATE SET score=excluded.score
                """,
                (str(node), affinity),
            )

    # Prepare mutual affinity data
    pairs: Dict[Tuple[str, str], Dict[str, float]] = {}

    for source, target, data in dal._graph.edges(data=True):
        await db.set_relationship(
            source,
            target,
            int(data.get("interaction_count", 0)),
            float(data.get("sentiment_sum", 0.0)),
            float(data.get("interaction_weight", 0.0)),
            data.get("last_interaction"),
        )
        a, b = sorted((source, target))
        info = pairs.setdefault((a, b), {"score": 0, "weight": 0.0, "last": 0.0})
        info["score"] += int(data.get("interaction_count", 0))
        info["weight"] += float(data.get("interaction_weight", 0.0))
        info["last"] = max(info["last"], float(data.get("last_interaction", 0.0)))

    for (a, b), info in pairs.items():
        score = info["score"] // 2
        weight = info["weight"] / 2
        last = info["last"] if info["last"] else None
        await db._db.execute(
            """
            INSERT INTO mutual_affinity (user_a, user_b, score, interaction_weight, last_interaction)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(user_a, user_b) DO UPDATE SET
                score=excluded.score,
                interaction_weight=excluded.interaction_weight,
                last_interaction=excluded.last_interaction
            """,
            (a, b, score, weight, last),
        )

    await db._db.commit()
    await db.close()


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_graph", help="Path to the legacy user_graph.json file")
    parser.add_argument("database", help="Destination SQLite database path")
    args = parser.parse_args()
    asyncio.run(migrate(args.json_graph, args.database))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    _main()

