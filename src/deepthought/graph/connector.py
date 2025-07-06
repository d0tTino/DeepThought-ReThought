"""Graph database connectors for Memgraph and Neo4j."""

from __future__ import annotations

from typing import Any, Dict, Optional

try:  # pragma: no cover - optional dependency
    from neo4j import GraphDatabase
except Exception:  # pragma: no cover - driver not installed
    GraphDatabase = None  # type: ignore[assignment]

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
        """Execute ``query`` and return the resulting rows as a list.

        If the underlying connection exposes an ``execute`` method (for
        example ``pymemgraph.Memgraph``), that method is used directly. In this
        case the connection is committed when possible and any results are
        collected into a list before being returned.
        """

        conn = self.connect()
        if hasattr(conn, "execute"):
            result = conn.execute(query, params or {})
            if hasattr(conn, "commit"):
                conn.commit()

            if hasattr(result, "fetchall"):
                return result.fetchall()
            if hasattr(conn, "fetchall"):
                return conn.fetchall()
            try:
                return list(result)
            except Exception:
                return [result] if result is not None else []

        cur = conn.cursor()
        try:
            cur.execute(query, params or {})
            rows = cur.fetchall()
            if hasattr(conn, "commit"):
                conn.commit()
            return rows
        finally:
            cur.close()


class Neo4jConnector:
    """Connector using the :mod:`neo4j` driver."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 7687,
        username: str = "neo4j",
        password: str = "neo4j",
    ) -> None:
        self._uri = f"bolt://{host}:{port}"
        self._auth = (username, password)
        self._driver: Optional[Any] = None

    def connect(self) -> Any:
        if not self._driver:
            if GraphDatabase is None:
                raise ImportError("neo4j-driver is not installed")
            self._driver = GraphDatabase.driver(self._uri, auth=self._auth)
        return self._driver

    def close(self) -> None:
        if self._driver:
            self._driver.close()
        self._driver = None

    def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> list:
        driver = self.connect()
        with driver.session() as session:
            result = session.run(query, params or {})
            try:
                return [record.data() for record in result]
            except Exception:  # pragma: no cover - defensive
                return list(result)
