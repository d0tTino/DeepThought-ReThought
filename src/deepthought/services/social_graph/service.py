"""JetStream-driven social graph service."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any, Dict, DefaultDict

from ...bus import Publisher, Subscriber
from ...eda.events import EventSubjects
from ..base import BaseService

logger = logging.getLogger(__name__)


class SocialGraphService(BaseService):
    """Consume graph update events and emit snapshots."""

    DURABLE_NAME = "sg_upd_v1"

    def __init__(self, subscriber: Subscriber, publisher: Publisher) -> None:
        super().__init__(subscriber, publisher)
        self._graph: DefaultDict[str, Dict[str, Any]] = defaultdict(lambda: {"edges": {}})
        self._start_lock = asyncio.Lock()
        self._started = False

    async def start(self) -> None:
        """Subscribe to JetStream subjects for incremental updates."""
        if self._started:
            return
        async with self._start_lock:
            if self._started:
                return
            await self._subscriber.subscribe(
                subject=EventSubjects.SOCIAL_GRAPH_UPDATE,
                handler=self._handle_update,
                use_jetstream=True,
                durable=self.DURABLE_NAME,
            )
            self._started = True
            logger.info("SocialGraphService listening on %s", EventSubjects.SOCIAL_GRAPH_UPDATE)

    async def _handle_update(self, msg: Any) -> None:
        payload = self._decode_message(msg)
        user_id = payload.get("user_id")
        if not user_id:
            logger.warning("Received social graph update without user_id: %s", payload)
            await self._ack_message(msg)
            return

        updates = payload.get("updates", {})
        edges = updates.get("edges", [])
        removals = updates.get("remove", [])

        user_graph = self._graph[user_id]
        edge_map: Dict[str, Any] = user_graph.setdefault("edges", {})
        for edge in edges:
            target = edge.get("target")
            if not target:
                continue
            weight = edge.get("weight", 1.0)
            edge_map[target] = weight
        for target in removals:
            edge_map.pop(target, None)

        snapshot = {
            "user_id": user_id,
            "graph": {"edges": edge_map.copy()},
            "timestamp": payload.get("timestamp"),
            "meta": {"source": "social_graph_service"},
        }

        logger.debug("Publishing graph snapshot for %s with %d edges", user_id, len(edge_map))
        await self._publish(EventSubjects.SOCIAL_GRAPH_SNAPSHOT, snapshot)
        await self._ack_message(msg)
