from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

from nats.aio.client import Client as NATS
from nats.aio.msg import Msg
from nats.js.client import JetStreamContext

from ..config import Settings, get_settings
from ..eda.contracts import EventEnvelope, decode_payload_or_envelope
from ..eda.events import (
    BDIIntentionPayload,
    EventSubjects,
    MemoryRetrievedPayload,
    PerceptionEmbeddingsPayload,
)
from ..memory import MemoryLifecyclePolicy, MemoryTier, create_memory_backend
from ..memory.fact_extractor import extract_typed_fact_triples_from_turn
from ..memory.graph import (
    CypherGraphMemoryStore,
    InMemoryGraphMemoryStore,
    ingest_conversation_turns,
    retrieve_topic_context,
    retrieve_user_context,
)
from ..memory.graph.pipeline import ingest_fact_triples
from ..fact_schema import format_fact_snippet, make_canonical_fact
from ..memory.graph.store import utc_now_iso
from ..memory.tiered import TieredMemory
from ..metrics.prometheus import INPUT_LATENCY_SECONDS, INPUTS_TOTAL
from ..search import OfflineSearch
from .base import BaseService
from .db_manager import DBManager
from .input_enrichment_service import InputEnrichmentService

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalPolicy:
    recent_turns: int = 4
    durable_user_facts: int = 4
    topic_memory: int = 4
    continuity_summaries: int = 2
    final_fact_budget: int = 12


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
        self._ensure_graph_backend_ready()
        self._db = db or DBManager()
        if search is None:
            db_path = self._settings.search_db
            if db_path:
                if not os.path.exists(db_path):
                    try:
                        search = OfflineSearch.create_index(db_path, [])
                    except ValueError:
                        logger.warning(
                            "No documents available for search index; disabling offline search"
                        )
                        search = None
                else:
                    search = OfflineSearch(db_path)
        self._search = search
        self._top_k = self._settings.memory_top_k
        self._input_enrichment = InputEnrichmentService()
        self._graph_memory = self._build_graph_memory_store()
        self._lifecycle_policy = MemoryLifecyclePolicy()
        self._tiered_turns: dict[
            str, list[dict[str, str | float | tuple[str, ...]]]
        ] = {
            MemoryTier.EPHEMERAL: [],
            MemoryTier.WORKING: [],
            MemoryTier.LONG_TERM: [],
        }
        self._salience_summaries: list[dict[str, str | float]] = []
        self._consolidation_task: asyncio.Task | None = None
        self._consolidation_interval_s = 60.0
        self._retrieval_policy = RetrievalPolicy()

    def _graph_connector(self):
        graph_backend = getattr(self._memory, "graph_backend", None)
        if hasattr(graph_backend, "_dal"):
            return getattr(getattr(graph_backend, "_dal", None), "_connector", None)
        if graph_backend is not None and hasattr(graph_backend, "_connector"):
            return getattr(graph_backend, "_connector", None)
        return None

    def _ensure_graph_backend_ready(self) -> None:
        backend = (self._settings.graph_backend or "").lower()
        if backend in {"stub", "noop", "none", "inmemory", "in-memory"}:
            if self._settings.runtime_profile != "test":
                raise RuntimeError("In-memory stub graph backend is only allowed in test profile")
            return
        if backend == "file":
            graph_file = getattr(self._settings, "graph_local_path", "graph_memory.json")
            parent = os.path.dirname(graph_file)
            if parent:
                os.makedirs(parent, exist_ok=True)
            try:
                with open(graph_file, "a", encoding="utf-8"):
                    pass
            except OSError as exc:
                raise RuntimeError(f"Local graph backend is unavailable: {graph_file}") from exc
            return
        connector = self._graph_connector()
        if connector is None:
            raise RuntimeError(f"Configured graph backend '{backend}' did not expose a connector")
        try:
            connector.execute("RETURN 1 AS ok", {})
        except Exception as exc:
            raise RuntimeError(
                f"Configured graph backend '{backend}' is unavailable for runtime profile "
                f"'{self._settings.runtime_profile}'"
            ) from exc

    def _build_graph_memory_store(self):
        backend = (self._settings.graph_backend or "").lower()
        if backend in {"memgraph", "neo4j"}:
            connector = self._graph_connector()
            if connector is None:
                raise RuntimeError(f"Configured graph backend '{backend}' did not expose a connector")
            return CypherGraphMemoryStore(connector)
        return InMemoryGraphMemoryStore()

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
                fact = make_canonical_fact(
                    subject=str(user_id),
                    predicate="memory_note",
                    object_value=str(memory),
                    provenance={"source": "db_context"},
                    confidence=0.6,
                    attributes={"topic": topic},
                )
                db_snippets.append(format_fact_snippet(fact))

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

    async def _db_context(
        self,
        resolved_user_id: str | int,
        channel_id: str | int | None = None,
    ) -> List[str]:
        if self._db is None or resolved_user_id is None:
            return []
        rows = await self._db.recall_user(resolved_user_id, limit=self._top_k)
        snippets = []
        for m in rows:
            if len(m) <= 1 or not m[1]:
                continue
            fact = make_canonical_fact(
                subject=str(resolved_user_id),
                predicate="memory_note",
                object_value=str(m[1]),
                provenance={"source": "db_context"},
                confidence=0.6,
                attributes={"topic": m[0]},
            )
            snippets.append(format_fact_snippet(fact))
        if channel_id is None:
            return snippets
        return snippets

    def _ingest_with_lifecycle(
        self, *, user_id: str, text: str, input_id: str, timestamp: str
    ) -> None:
        scored = self._lifecycle_policy.score_event(text)
        bucket = self._tiered_turns[scored.tier]
        bucket.append(
            {
                "user_id": str(user_id),
                "text": text,
                "input_id": input_id,
                "timestamp": timestamp,
                "salience": scored.salience,
                "reasons": scored.reason_tags,
            }
        )
        if scored.tier is MemoryTier.WORKING and len(bucket) > max(self._top_k * 3, 16):
            del bucket[: -max(self._top_k * 3, 16)]

    def run_consolidation_cycle(self) -> int:
        archived = 0
        stale_ephemeral = self._tiered_turns[MemoryTier.EPHEMERAL]
        while stale_ephemeral and len(stale_ephemeral) > max(self._top_k, 8):
            entry = stale_ephemeral.pop(0)
            note = str(entry.get("text", ""))
            self._salience_summaries.append(
                {
                    "summary": note[:220],
                    "salience": float(entry.get("salience", 0.1)),
                    "timestamp": str(entry.get("timestamp") or utc_now_iso()),
                    "source_tier": MemoryTier.EPHEMERAL,
                    "user_id": str(entry.get("user_id") or "anonymous"),
                }
            )
            archived += 1
        if len(self._salience_summaries) > 200:
            self._salience_summaries = self._salience_summaries[-200:]
        return archived

    async def _consolidation_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._consolidation_interval_s)
                archived = self.run_consolidation_cycle()
                if archived:
                    logger.debug("Consolidated %s low-salience turns", archived)
        except asyncio.CancelledError:
            return

    def _prioritized_graph_facts(self, user_id: str, topic: str) -> list[str]:
        summary_evidence = [
            item
            for item in self._salience_summaries
            if (
                item.get("user_id") == str(user_id)
                and topic.lower() in str(item.get("summary", "")).lower()
            )
            or item.get("user_id") == str(user_id)
        ]
        summary_evidence = sorted(
            summary_evidence,
            key=lambda e: (float(e.get("salience", 0.0)), str(e.get("timestamp", ""))),
            reverse=True,
        )
        summary_facts = [
            f"summary: {item['summary']}" for item in summary_evidence[: self._top_k]
        ]

        graph_user_evidence = retrieve_user_context(
            self._graph_memory, str(user_id), limit=self._top_k
        )
        graph_topic_evidence = retrieve_topic_context(
            self._graph_memory, topic, limit=self._top_k
        )
        graph_facts = []
        for ev in [*graph_user_evidence, *graph_topic_evidence]:
            fact = make_canonical_fact(
                subject=str(ev.attributes.get("subject") or user_id),
                predicate=str(ev.relation_type or "fact"),
                object_value=str(ev.summary.split(":", 1)[-1].strip() if ":" in ev.summary else ev.summary),
                object_id=ev.entity_id,
                provenance={"source": ev.provenance.source, "source_id": ev.provenance.source_id, "observed_at": ev.provenance.observed_at},
                confidence=ev.confidence,
                created_at=str(ev.attributes.get("created_at") or ev.attributes.get("timestamp") or utc_now_iso()),
                updated_at=str(ev.attributes.get("updated_at") or ev.attributes.get("timestamp") or utc_now_iso()),
                attributes=dict(ev.attributes),
            )
            graph_facts.append(format_fact_snippet(fact))
        return summary_facts + graph_facts

    @staticmethod
    def _normalize_fact_entries(entries: list[str], *, prefix: str | None = None) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for item in entries:
            text = " ".join(str(item).split()).strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(f"{prefix}{text}" if prefix else text)
        return normalized

    def _recent_episodic_turns(
        self,
        *,
        conversation_window: list[dict[str, object]],
        user_input: str,
    ) -> list[str]:
        prior_turns: list[str] = []
        for turn in conversation_window[:-1]:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("role") or "user")
            text = " ".join(str(turn.get("text") or "").split()).strip()
            if text:
                prior_turns.append(f"{role}: {text}")
        if not prior_turns and user_input.strip():
            prior_turns.append(f"user: {user_input.strip()}")
        return self._normalize_fact_entries(prior_turns)[-self._retrieval_policy.recent_turns :]

    def _channel_thread_continuity(
        self,
        *,
        conversation_window: list[dict[str, object]],
        recent_turn_summary: str | None,
        channel_id: str | None,
        thread_id: str | None,
    ) -> list[str]:
        continuity: list[str] = []
        if recent_turn_summary and recent_turn_summary.strip():
            continuity.append(f"recent summary: {recent_turn_summary.strip()}")
        if thread_id:
            continuity.append(f"thread focus: thread:{thread_id}")
        elif channel_id:
            continuity.append(f"channel focus: channel:{channel_id}")
        if conversation_window:
            participants = sorted(
                {
                    str(turn.get("author_id")).strip()
                    for turn in conversation_window
                    if isinstance(turn, dict) and str(turn.get("author_id") or "").strip()
                }
            )
            if participants:
                continuity.append(f"active participants: {', '.join(participants[:4])}")
        return self._normalize_fact_entries(continuity)[: self._retrieval_policy.continuity_summaries]

    def _assemble_retrieval_layers(
        self,
        *,
        recent_turns: list[str],
        durable_user_facts: list[str],
        topic_memory: list[str],
        continuity_summaries: list[str],
    ) -> dict[str, list[str]]:
        return {
            "recent_episodic_turns": recent_turns[: self._retrieval_policy.recent_turns],
            "durable_user_facts": durable_user_facts[: self._retrieval_policy.durable_user_facts],
            "topic_entity_memory": topic_memory[: self._retrieval_policy.topic_memory],
            "channel_thread_continuity": continuity_summaries[: self._retrieval_policy.continuity_summaries],
        }

    def _flatten_retrieval_layers(self, layers: dict[str, list[str]]) -> list[str]:
        ordered: list[str] = []
        for layer_name in (
            "recent_episodic_turns",
            "durable_user_facts",
            "topic_entity_memory",
            "channel_thread_continuity",
        ):
            for item in layers.get(layer_name, []):
                ordered.append(f"[{layer_name}] {item}")
        return ordered[: self._retrieval_policy.final_fact_budget]

    async def _handle_input(self, msg: Msg) -> None:
        input_id = "unknown"
        start = time.perf_counter()
        try:
            raw_data = json.loads(msg.data.decode())
            if not isinstance(raw_data, dict):
                raise ValueError("InputReceived payload must be a dict")
            decoded_payload, envelope_meta = decode_payload_or_envelope(
                EventSubjects.MEMORY_RETRIEVAL_REQUESTED, raw_data
            )
            enriched = self._input_enrichment.parse_input_received_data(
                decoded_payload, headers=getattr(msg, "headers", None)
            )
            input_id = enriched.input_id
            user_input = enriched.user_input
            user_id = enriched.user_id
            author_id = enriched.author_id
            channel_id = enriched.channel_id
            conversation_window = decoded_payload.get("conversation_window")
            if not isinstance(conversation_window, list):
                conversation_window = []
            recent_turn_summary = decoded_payload.get("recent_turn_summary")
            if not isinstance(recent_turn_summary, str):
                recent_turn_summary = None
            thread_id = decoded_payload.get("thread_id")
            if not isinstance(thread_id, str):
                thread_id = None
            logger.info("CognitiveCoreService received input %s", input_id)

            resolved_user_id = enriched.resolved_user_id
            turn_timestamp = datetime.now(timezone.utc).isoformat()
            self._memory.store_interaction(user_input)
            self._ingest_with_lifecycle(
                user_id=str(resolved_user_id),
                text=user_input,
                input_id=input_id,
                timestamp=turn_timestamp,
            )
            scored = self._lifecycle_policy.score_event(user_input)
            db_topic = f"tier:{scored.tier}"
            await self._db.store_memory(resolved_user_id, user_input, topic=db_topic)

            ingest_conversation_turns(
                [
                    {
                        "user_id": resolved_user_id,
                        "text": user_input,
                        "timestamp": turn_timestamp,
                        "input_id": input_id,
                    }
                ],
                self._graph_memory,
                default_user_id=str(resolved_user_id),
            )

            extracted_triples = extract_typed_fact_triples_from_turn(
                user_id=str(resolved_user_id),
                message=user_input,
                timestamp=turn_timestamp,
                source_id=input_id,
            )
            if extracted_triples:
                enriched_triples = []
                for triple in extracted_triples:
                    attrs = dict(triple.get("attributes", {}))
                    attrs.setdefault("memory_tier", str(scored.tier))
                    attrs.setdefault("salience", scored.salience)
                    if scored.reason_tags:
                        attrs.setdefault("salience_reasons", list(scored.reason_tags))
                    enriched_triples.append({**triple, "attributes": attrs})
                ingest_fact_triples(
                    enriched_triples,
                    self._graph_memory,
                    timestamp=turn_timestamp,
                    source_id=input_id,
                )

            memory_facts = self._memory.retrieve_context(user_input) if self._memory else []
            vector_facts: List[str] = []
            if self._search:
                try:
                    vector_facts = self._search.search(user_input, limit=self._top_k)
                except Exception:  # pragma: no cover - defensive
                    logger.error("Offline search failed", exc_info=True)

            self.run_consolidation_cycle()
            db_facts = await self._db_context(
                resolved_user_id=resolved_user_id, channel_id=channel_id
            )
            graph_facts = self._prioritized_graph_facts(
                str(resolved_user_id), user_input
            )

            recent_turns = self._recent_episodic_turns(
                conversation_window=conversation_window,
                user_input=user_input,
            )
            durable_user_facts = self._normalize_fact_entries(
                [*db_facts, *memory_facts, *vector_facts]
            )
            topic_memory = self._normalize_fact_entries(graph_facts)
            continuity_summaries = self._channel_thread_continuity(
                conversation_window=conversation_window,
                recent_turn_summary=recent_turn_summary,
                channel_id=channel_id,
                thread_id=thread_id,
            )
            retrieval_layers = self._assemble_retrieval_layers(
                recent_turns=recent_turns,
                durable_user_facts=durable_user_facts,
                topic_memory=topic_memory,
                continuity_summaries=continuity_summaries,
            )
            facts = self._flatten_retrieval_layers(retrieval_layers)
            trace_id = (
                envelope_meta.get("trace_id")
                if isinstance(envelope_meta.get("trace_id"), str)
                else None
            )
            causation_id = (
                envelope_meta.get("event_id")
                if isinstance(envelope_meta.get("event_id"), str)
                else input_id
            )
            payload = MemoryRetrievedPayload(
                retrieved_knowledge={
                    "facts": facts,
                    "source": "cognitive_core",
                    "layers": retrieval_layers,
                    "retrieval_policy": self._retrieval_policy.__dict__,
                    "conversation_window": conversation_window[-self._retrieval_policy.recent_turns :],
                    "recent_turn_summary": recent_turn_summary,
                },
                user_input=user_input,
                input_id=input_id,
                user_id=author_id or user_id,
                author_id=author_id,
                channel_id=channel_id,
                recent_turn_summary=recent_turn_summary,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

            envelope = EventEnvelope.build(
                subject=EventSubjects.MEMORY_RETRIEVED,
                payload=json.loads(payload.to_json()),
                producer=self.__class__.__name__,
                trace_id=trace_id,
                causation_id=causation_id,
            )
            await self._publisher.publish(
                EventSubjects.MEMORY_RETRIEVED,
                envelope.__dict__,
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
            INPUT_LATENCY_SECONDS.labels(service="cognitive_core_service").observe(
                duration
            )

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
                        new_vectors = [
                            v for v, _id in zip(vectors, ids) if _id in missing
                        ]
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
            logger.error("Failed to handle perception embedding event", exc_info=True)
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
            logger.info(
                "CognitiveCoreService received intention goal: %s", payload.goal
            )
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
            subject=EventSubjects.MEMORY_RETRIEVAL_REQUESTED,
            handler=self._handle_input,
            use_jetstream=True,
            durable=durable_name,
        )
        for subject, suffix in (
            (EventSubjects.PERCEPTION_EMBEDDINGS, "perception_fused"),
            (EventSubjects.PERCEPTION_IMAGE_EMBED, "perception_image"),
            (EventSubjects.PERCEPTION_AUDIO_EMBED, "perception_audio"),
            (EventSubjects.PERCEPTION_VIDEO_EMBED, "perception_video"),
        ):
            self.add_subscription(
                subject=subject,
                handler=self._handle_embeddings,
                use_jetstream=True,
                durable=f"{durable_name}_{suffix}",
            )
        self.add_subscription(
            subject=EventSubjects.BDI_INTENTION,
            handler=self._handle_intention,
            use_jetstream=True,
            durable=f"{durable_name}_bdi",
        )
        self._ensure_graph_backend_ready()
        started = await super().start()
        if started:
            logger.info(
                "CognitiveCoreService subscribed to %s",
                EventSubjects.MEMORY_RETRIEVAL_REQUESTED,
            )
            if self._consolidation_task is None or self._consolidation_task.done():
                self._consolidation_task = asyncio.create_task(
                    self._consolidation_loop()
                )
        return started

    async def stop(self) -> None:
        if self._consolidation_task and not self._consolidation_task.done():
            self._consolidation_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._consolidation_task
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
