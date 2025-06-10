"""Minimal Memgraph connector using the :mod:`pymemgraph` driver."""

from __future__ import annotations

from typing import Any, Dict, Optional

try:  # pragma: no cover - optional dependency
    from pymemgraph import Memgraph
except Exception:  # pragma: no cover - driver not installed
    Memgraph = None  # type: ignore[assignment]


class GraphConnector:
    """Wrapper around :mod:`pymemgraph` to execute Cypher queries."""

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
        self._connection: Optional[Any] = None

    def connect(self) -> Any:
        """Establish connection if not already connected."""
        if not self._connection:
            if Memgraph is None:
                raise ImportError("pymemgraph is not installed")
            self._connection = Memgraph(**self._params)
        return self._connection

    def close(self) -> None:
        if self._connection and hasattr(self._connection, "close"):
            self._connection.close()
        self._connection = None

    def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> list:
        conn = self.connect()
        if hasattr(conn, "execute"):
            return conn.execute(query, params or {})
        cur = conn.cursor()
        cur.execute(query, params or {})
        rows = cur.fetchall()
        if hasattr(cur, "close"):
            cur.close()
        return rows
