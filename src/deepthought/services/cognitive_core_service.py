from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import List, Optional

from nats.aio.client import Client as NATS
from nats.aio.msg import Msg
from nats.js.client import JetStreamContext

from ..config import Settings, get_settings
from ..eda.events import (
    BDIIntentionPayload,
    EventSubjects,
    MemoryRetrievedPayload,
    PerceptionEmbeddingsPayload,
)
from ..memory import create_memory_backend
from ..memory.tiered import TieredMemory
from ..metrics.prometheus import INPUT_LATENCY_SECONDS, INPUTS_TOTAL
from ..perception.social_perception import analyze as analyze_social
from ..search import OfflineSearch
from .base import BaseService
from .db_manager import DBManager

logger = logging.getLogger(__name__)


class CognitiveCoreService(BaseService):
    """Unified service for vector, graph and relational memory."""

    def __init__(
        self,
        nats_client: Optional[NATS] = None,
        js_context: Optional[JetStreamContext] = None,
        settings: Settings | None = None,
        memory: Optional[TieredMemory] = None,
        db: Optional[DBManager] = None,
        search: OfflineSearch | None = None,
        *,
        nats_url: str | None = None,
        connect_retries: int = 1,
        connect_timeout: float = 2.0,
    ) -> None:
        super().__init__(
            nats_client,
            js_context,
            nats_url=nats_url,
            connect_retries=connect_retries,
            connect_timeout=connect_timeout,
        )
        self._settings = settings or get_settings()
        if memory is None:
            memory = create_memory_backend(settings=self._settings)
        self._memory = memory
        self._db = db or DBManager()
        if search is None:
            db_path = self._settings.search_db
            if db_path:
                if not os.path.exists(db_path):
                    try:
                        search = OfflineSearch.create_index(db_path, [])
                    except ValueError:
                        logger.warning("No documents available for search index; disabling offline search")
                        search = None
                else:
                    search = OfflineSearch(db_path)
        self._search = search
        self._top_k = self._settings.memory_top_k

    @classmethod
    def from_config(
        cls,
        nats_client: Optional[NATS] = None,
        js_context: Optional[JetStreamContext] = None,
        settings: Settings | None = None,
        *,
        nats_url: str | None = None,
        connect_retries: int = 1,
        connect_timeout: float = 2.0,
    ) -> "CognitiveCoreService":
        """Return an instance configured from ``Settings`` or environment."""

        cfg = settings or get_settings()
        memory = create_memory_backend(settings=cfg)
        db = DBManager(cfg.social_graph_db)
        return cls(
            nats_client,
            js_context,
            cfg,
            memory=memory,
            db=db,
            nats_url=nats_url,
            connect_retries=connect_retries,
            connect_timeout=connect_timeout,
        )

    def retrieve_context(self, prompt: str) -> List[str]:
        memory_facts = self._memory.retrieve_context(prompt) if self._memory else []
        search_facts: List[str] = []
        if self._search:
            try:
                search_facts = self._search.search(prompt, limit=self._top_k)
            except Exception:  # pragma: no cover - defensive
                logger.error("Offline search failed", exc_info=True)
        merged: List[str] = []
        seen = set()
        for item in memory_facts + search_facts:
            if item not in seen:
                seen.add(item)
                merged.append(item)
        return merged

    async def retrieve_user_facts(
        self,
        user_id: str | int,
        channel_id: str | int | None = None,
        *,
        prompt: str | None = None,
        limit: int | None = None,
    ) -> List[str]:
        """Return top facts/snippets for ``user_id`` and ``channel_id``."""

        max_items = self._top_k if limit is None else int(limit)
        if max_items <= 0:
            return []

        seed_prompt = prompt
        if not seed_prompt:
            seed_parts = []
            if user_id is not None:
                seed_parts.append(f"user:{user_id}")
            if channel_id is not None:
                seed_parts.append(f"channel:{channel_id}")
            seed_prompt = " ".join(seed_parts)

        memory_facts = self.retrieve_context(seed_prompt) if seed_prompt else []
        db_snippets: List[str] = []
        if self._db is not None and user_id is not None:
            rows = await self._db.recall_user(user_id, limit=max_items)
            for topic, memory in rows:
                if not memory or not str(memory).strip():
                    continue
                entry = f"[{topic}] {memory}" if topic else str(memory)
                db_snippets.append(entry.strip())

        merged: List[str] = []
        seen = set()
        for item in memory_facts + db_snippets:
            if item in seen:
                continue
            seen.add(item)
            merged.append(item)
            if len(merged) >= max_items:
                break
        return merged

    async def _db_context(self) -> List[str]:
        rows = await self._db.recall_user("user", limit=self._top_k)
        return [m[1] for m in rows]

    async def _handle_input(self, msg: Msg) -> None:
        input_id = "unknown"
        start = time.perf_counter()
        try:
            data = json.loads(msg.data.decode())
            if not isinstance(data, dict):
                raise ValueError("InputReceived payload must be a dict")
            input_id = data.get("input_id")
            user_input = data.get("user_input")
            headers = getattr(msg, "headers", None)
            user_id = data.get("user_id") or (headers.get("user_id") if headers else None)
            if not isinstance(user_id, str):
                user_id = None
            if not isinstance(input_id, str) or not isinstance(user_input, str):
                raise ValueError("Invalid input payload fields")
            logger.info("CognitiveCoreService received input %s", input_id)

            self._memory.store_interaction(user_input)
            await self._db.store_memory("user", user_input)
            await self._db.log_interaction("user", None)
            try:
                perception = analyze_social(user_input)
            except Exception as e:  # pragma: no cover - defensive
                logger.error("Failed to analyze social perception: %s", e, exc_info=True)
                perception = {"flirtation": 0.0, "avoidance": 0.0, "manipulation": 0.0}
            await self._db.store_memory("user", json.dumps(perception), topic="social_perception")
            delta = perception.get("flirtation", 0.0) - (
                perception.get("avoidance", 0.0) + perception.get("manipulation", 0.0)
            )
            await self._db.adjust_affinity("user", delta)

            mem_facts = self.retrieve_context(user_input)
            db_facts = await self._db_context()
            facts: List[str] = []
            seen = set()
            for item in mem_facts + db_facts:
                if item not in seen:
                    seen.add(item)
                    facts.append(item)
            payload = MemoryRetrievedPayload(
                retrieved_knowledge={"facts": facts, "source": "cognitive_core"},
                input_id=input_id,
                user_id=user_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

            await self._publisher.publish(
                EventSubjects.MEMORY_RETRIEVED,
                payload,
                use_jetstream=True,
                timeout=10.0,
            )
            logger.info("CognitiveCoreService published memory event %s", input_id)
            if hasattr(msg, "ack") and callable(msg.ack):
                try:
                    await msg.ack()
                except Exception:
                    logger.error("Failed to ack message", exc_info=True)
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("Invalid InputReceived payload: %s", e, exc_info=True)
            if hasattr(msg, "nak") and callable(msg.nak):
                try:
                    await msg.nak()
                except Exception:
                    logger.error("Failed to NAK message", exc_info=True)
            elif hasattr(msg, "ack") and callable(msg.ack):
                try:
                    await msg.ack()
                except Exception:
                    logger.error("Failed to ack message after error", exc_info=True)
        except Exception as e:  # pragma: no cover - defensive
            logger.error("Error in CognitiveCoreService handler: %s", e, exc_info=True)
            if hasattr(msg, "nak") and callable(msg.nak):
                try:
                    await msg.nak()
                except Exception:
                    logger.error("Failed to NAK message", exc_info=True)
            elif hasattr(msg, "ack") and callable(msg.ack):
                try:
                    await msg.ack()
                except Exception:
                    logger.error("Failed to ack message after error", exc_info=True)
        finally:
            duration = time.perf_counter() - start
            INPUTS_TOTAL.labels(service="cognitive_core_service").inc()
            INPUT_LATENCY_SECONDS.labels(service="cognitive_core_service").observe(duration)

    async def _handle_embeddings(self, msg: Msg) -> None:
        """Store perception embeddings in the vector store and knowledge graph."""
        try:
            payload = PerceptionEmbeddingsPayload.from_json(msg.data.decode())
            message_id = str(payload.message_id)
            store = getattr(self._memory, "_store", None)
            graph = getattr(self._memory, "graph_backend", None)

            # Upsert vectors for each span keyed by (message_id, span_index)
            if store and hasattr(store, "upsert_vectors"):
                vectors = payload.fused
                # Fall back to the first modality if fused embeddings are unavailable
                if vectors is None and payload.by_modality:
                    first_mod = next(iter(payload.by_modality.values()))
                    vectors = first_mod.embeddings
                if vectors:
                    ids = [f"{message_id}:{idx}" for idx in range(len(vectors))]
                    missing = getattr(store, "missing_ids", lambda x: list(x))(ids)
                    if missing:
                        new_vectors = [v for v, _id in zip(vectors, ids) if _id in missing]
                        store.upsert_vectors(new_vectors, missing)

            # Insert nodes/edges in the KG with modality and timestamp metadata
            if graph:
                timestamp = datetime.now(timezone.utc).isoformat()
                for modality, mod in payload.by_modality.items():
                    for idx, span in enumerate(mod.spans):
                        span_id = f"{message_id}:{idx}"
                        try:
                            exists = graph.query_subgraph(
                                "MATCH (:Message {id: $mid})-[:HAS_SPAN]->(:Span {id: $sid}) RETURN 1",
                                {"mid": message_id, "sid": span_id},
                            )
                            if exists:
                                continue
                        except Exception:  # pragma: no cover - defensive
                            pass
                        vector = None
                        if payload.fused and idx < len(payload.fused):
                            vector = payload.fused[idx]
                        elif idx < len(mod.embeddings):
                            vector = mod.embeddings[idx]
                        graph.query_subgraph(
                            "MERGE (m:Message {id: $mid}) "
                            "MERGE (s:Span {id: $sid}) "
                            "SET s.embedding = $embedding, s.start = $start, s.end = $end, "
                            "s.modality = $modality, s.timestamp = $timestamp "
                            "MERGE (m)-[:HAS_SPAN {modality: $modality, timestamp: $timestamp}]->(s)",
                            {
                                "mid": message_id,
                                "sid": span_id,
                                "embedding": vector,
                                "start": span[0],
                                "end": span[1],
                                "modality": modality,
                                "timestamp": timestamp,
                            },
                        )

            if hasattr(msg, "ack") and callable(msg.ack):
                await msg.ack()
        except Exception:
            logger.error("Failed to handle PERCEPTION_EMBEDDINGS", exc_info=True)
            if hasattr(msg, "nak") and callable(msg.nak):
                try:
                    await msg.nak()
                except Exception:
                    logger.error("Failed to NAK message", exc_info=True)
            elif hasattr(msg, "ack") and callable(msg.ack):
                try:
                    await msg.ack()
                except Exception:
                    logger.error("Failed to ack message after error", exc_info=True)

    async def _handle_intention(self, msg: Msg) -> None:
        """React to BDI_INTENTION events by logging the goal."""
        try:
            payload = BDIIntentionPayload.from_json(msg.data.decode())
            logger.info("CognitiveCoreService received intention goal: %s", payload.goal)
            if hasattr(msg, "ack") and callable(msg.ack):
                await msg.ack()
        except Exception:
            logger.error("Failed to handle BDI_INTENTION", exc_info=True)
            if hasattr(msg, "nak") and callable(msg.nak):
                try:
                    await msg.nak()
                except Exception:
                    logger.error("Failed to NAK message", exc_info=True)
            elif hasattr(msg, "ack") and callable(msg.ack):
                try:
                    await msg.ack()
                except Exception:
                    logger.error("Failed to ack message after error", exc_info=True)

    async def start(self, durable_name: str = "cognitive_core_listener") -> bool:
        self._subscriptions.clear()
        self.add_subscription(
            subject=EventSubjects.INPUT_RECEIVED,
            handler=self._handle_input,
            use_jetstream=True,
            durable=durable_name,
        )
        self.add_subscription(
            subject=EventSubjects.PERCEPTION_EMBEDDINGS,
            handler=self._handle_embeddings,
            use_jetstream=True,
            durable=f"{durable_name}_perception",
        )
        self.add_subscription(
            subject=EventSubjects.BDI_INTENTION,
            handler=self._handle_intention,
            use_jetstream=True,
            durable=f"{durable_name}_bdi",
        )
        started = await super().start()
        if started:
            logger.info("CognitiveCoreService subscribed to %s", EventSubjects.INPUT_RECEIVED)
        return started

    async def stop(self) -> None:
        try:
            backend = getattr(self._memory, "graph_backend", None)
            connector = None
            if hasattr(backend, "_dal"):
                connector = getattr(getattr(backend, "_dal", None), "_connector", None)
            elif backend is not None:
                connector = getattr(backend, "_connector", None)
            if connector and hasattr(connector, "close"):
                connector.close()
        except Exception:
            logger.error("Failed to close graph connector", exc_info=True)
        if hasattr(self._db, "close"):
            try:
                await self._db.close()
            except Exception:
                logger.error("Failed to close DB connection", exc_info=True)
        await super().stop()

    async def __aenter__(self) -> "CognitiveCoreService":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.stop()
