"""Graph database connectors for Memgraph and Neo4j."""

from __future__ import annotations

import os
import time
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
        host: str | None = None,
        port: int | None = None,
        username: str | None = None,
        password: str | None = None,
        *,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        from ..config import get_settings

        settings = get_settings()

        host = host or settings.mg_host
        port = port or settings.mg_port

        if not host:
            raise ValueError("Memgraph host is required")
        if port in (None, ""):
            raise ValueError("Memgraph port is required")

        try:
            port_int = int(port)
        except (TypeError, ValueError) as exc:
            raise ValueError("Memgraph port must be an integer") from exc
        if port_int <= 0:
            raise ValueError("Memgraph port must be positive")

        self._params = {
            "host": host,
            "port": port_int,
            "username": username or settings.mg_user,
            "password": password or settings.mg_password,
        }
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._connection: Optional[Any] = None

    def connect(self) -> Any:
        """Establish connection if not already connected."""
        if not self._connection:
            if Memgraph is None:
                raise ImportError("pymemgraph is not installed")
            last_error: Exception | None = None
            for _ in range(max(1, self._max_retries)):
                try:
                    self._connection = Memgraph(**self._params)
                    break
                except Exception as exc:  # pragma: no cover - defensive
                    last_error = exc
                    time.sleep(self._retry_delay)
            if self._connection is None and last_error is not None:
                raise last_error
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
        host: str | None = None,
        port: int | None = None,
        username: str | None = None,
        password: str | None = None,
        *,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        from ..config import get_settings

        settings = get_settings()
        host = host or settings.neo4j_host
        port = int(port or settings.neo4j_port)
        username = username or settings.neo4j_user
        password = password or settings.neo4j_password
        self._uri = f"bolt://{host}:{port}"
        self._auth = (username, password)
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._driver: Optional[Any] = None

    def connect(self) -> Any:
        if not self._driver:
            if GraphDatabase is None:
                raise ImportError("neo4j-driver is not installed")
            last_error: Exception | None = None
            for _ in range(max(1, self._max_retries)):
                try:
                    self._driver = GraphDatabase.driver(self._uri, auth=self._auth)
                    break
                except Exception as exc:  # pragma: no cover - defensive
                    last_error = exc
                    time.sleep(self._retry_delay)
            if self._driver is None and last_error is not None:
                raise last_error
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
