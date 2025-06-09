"""Minimal Memgraph connector using pymgclient."""

from __future__ import annotations

from typing import Any, Dict, Optional
import mgclient


class GraphConnector:
    """Wrapper around mgclient to execute Cypher queries."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 7687,
        username: str = "",
        password: str = "",
    ) -> None:
        self._params = {
            "host": host,
            "port": port,
            "username": username,
            "password": password,
        }
        self._connection: Optional[mgclient.Connection] = None

    def connect(self) -> mgclient.Connection:
        """Establish connection if not already connected."""
        if not self._connection:
            self._connection = mgclient.connect(**self._params)
        return self._connection

    def close(self) -> None:
        if self._connection:
            self._connection.close()
            self._connection = None

    def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> list:
        conn = self.connect()
        cur = conn.cursor()
        cur.execute(query, params or {})
        rows = cur.fetchall()
        cur.close()
        return rows
