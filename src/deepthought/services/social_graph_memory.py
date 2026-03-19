"""High level helpers for recording and querying social interactions."""

from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Optional

from ..config import get_settings
from ..fact_schema import make_canonical_fact
from ..memory.graph import retrieve_user_context
from ..memory.graph.adapters import CypherGraphMemoryStore, InMemoryGraphMemoryStore
from ..memory.graph.store import GraphFact, GraphMemoryStore
from .prism_adapter import PrismEvent

try:  # Optional dependency
    from textblob import TextBlob  # type: ignore
except Exception:  # pragma: no cover - dependency missing
    TextBlob = None  # type: ignore

from .db_manager import DBManager

logger = logging.getLogger(__name__)

FRIEND_THRESHOLD = 0.5
RIVAL_THRESHOLD = -0.5
MIN_INTERACTIONS = 3

POSITIVE_EMOJIS = {"❤️", "👍"}
NEGATIVE_EMOJIS = {"💔", "👎"}
LATENCY_THRESHOLD = 5.0
SOCIAL_MODEL_VERSION = "v2"


class SocialGraphMemory:
    """Record messages and persist durable social-user model state."""

    def __init__(self, db_manager: Optional[DBManager] = None) -> None:
        self._db = db_manager or DBManager()
        self._graph_memory = self._build_graph_memory_store()

    def _build_graph_memory_store(self) -> GraphMemoryStore:
        settings = get_settings()
        backend = (settings.graph_backend or "").lower()
        if backend in {"memgraph", "neo4j"}:
            graph_backend = getattr(getattr(self._db, "_memory", None), "graph_backend", None)
            connector = getattr(graph_backend, "_connector", None)
            if connector is not None:
                return CypherGraphMemoryStore(connector)
        return InMemoryGraphMemoryStore()

    @staticmethod
    def _clamp(value: float, *, minimum: float = 0.0, maximum: float = 1.0) -> float:
        return max(minimum, min(maximum, float(value)))

    @staticmethod
    def _blend(previous: float, observed: float, *, weight: float = 0.35) -> float:
        return (1.0 - weight) * previous + weight * observed

    @staticmethod
    def _tier(score: float) -> str:
        if score >= 0.72:
            return "high"
        if score >= 0.38:
            return "medium"
        return "low"

    async def _load_social_model(self, user_id: str) -> dict[str, Any]:
        profile = await self._db.get_user_profile(user_id)
        if isinstance(profile, dict):
            social_model = profile.get("social_model")
            if isinstance(social_model, dict):
                return social_model
        return {
            "version": SOCIAL_MODEL_VERSION,
            "dimensions": {},
            "channel_specific_norms": {},
        }

    async def _save_social_model(self, user_id: str, social_model: dict[str, Any]) -> None:
        profile = await self._db.get_user_profile(user_id)
        merged = dict(profile) if isinstance(profile, dict) else {}
        merged["social_model"] = social_model
        await self._db.set_user_profile(user_id, merged)

    async def _upsert_social_fact(
        self,
        *,
        user_id: str,
        predicate: str,
        object_value: str,
        confidence: float,
        attributes: dict[str, Any],
    ) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        fact = make_canonical_fact(
            subject=str(user_id),
            predicate=predicate,
            object_value=object_value,
            provenance={"source": "social_graph_memory", "observed_at": timestamp},
            confidence=confidence,
            created_at=timestamp,
            updated_at=timestamp,
            attributes=attributes,
        )
        self._graph_memory.upsert_fact(GraphFact(**fact.__dict__))

    async def record_message(self, source: str, text: str, target: Optional[str] = None) -> None:
        try:
            score = float(TextBlob(text).sentiment.polarity) if TextBlob else 0.0
        except Exception:  # pragma: no cover - TextBlob failure
            logger.exception("Sentiment analysis failed")
            score = 0.0
        await self._db.log_interaction(source, target, sentiment_score=score)
        if target is not None:
            await self._update_relationship_type(source, target)

    async def get_affinity(self, user_id: str) -> int:
        return await self._db.get_affinity(user_id)

    async def get_friendliness(self, source: str, target: str) -> float:
        return await self._db.get_friendliness(source, target)

    async def get_hostility(self, source: str, target: str) -> float:
        return await self._db.get_hostility(source, target)

    async def get_mutual_affinity(self, user_a: str, user_b: str) -> float:
        return await self._db.get_pair_mutual_affinity(user_a, user_b)

    async def set_personality(self, user_id: str, traits) -> None:
        await self._db.set_user_profile(user_id, traits)

    async def get_personality(self, user_id: str):
        return await self._db.get_user_profile(user_id)

    async def get_relationship_stats(self, user_a: str, user_b: str) -> dict:
        ab = await self._db.get_relationship(user_a, user_b)
        ba = await self._db.get_relationship(user_b, user_a)

        def _stats(row: tuple | None) -> dict:
            if not row:
                return {
                    "count": 0,
                    "sentiment_sum": 0.0,
                    "avg_sentiment": 0.0,
                    "interaction_weight": 0.0,
                    "last_interaction": None,
                }
            count, sentiment_sum, weight, last = row
            avg = float(sentiment_sum) / count if count else 0.0
            return {
                "count": int(count),
                "sentiment_sum": float(sentiment_sum),
                "avg_sentiment": avg,
                "interaction_weight": float(weight),
                "last_interaction": last,
            }

        stats_a = _stats(ab)
        stats_b = _stats(ba)
        mutual = await self._db.get_pair_mutual_affinity(user_a, user_b)
        return {"pair": (user_a, user_b), "a_to_b": stats_a, "b_to_a": stats_b, "mutual_affinity": int(mutual)}

    async def _update_relationship_type(self, user_a: str, user_b: str) -> None:
        ab = await self._db.get_relationship(user_a, user_b) or (0, 0.0, 0.0, None)
        ba = await self._db.get_relationship(user_b, user_a) or (0, 0.0, 0.0, None)
        total_count = int(ab[0] or 0) + int(ba[0] or 0)
        total_sentiment = float(ab[1] or 0.0) + float(ba[1] or 0.0)
        status = "neutral"
        if total_count >= MIN_INTERACTIONS and total_count:
            avg = total_sentiment / total_count
            if avg >= FRIEND_THRESHOLD:
                status = "friend"
            elif avg <= RIVAL_THRESHOLD:
                status = "rival"
        await self._db.set_relationship_type(user_a, user_b, status)
        if status == "friend":
            await self._db.update_edge(user_a, user_b, "ally", 1.0)
        elif status == "rival":
            await self._db.update_edge(user_a, user_b, "rival", 1.0)

    async def update_relationship_type(self, user_a: str, user_b: str) -> None:
        await self._update_relationship_type(user_a, user_b)

    async def get_relationship_status(self, user_a: str, user_b: str) -> str | None:
        return await self._db.get_relationship_type(user_a, user_b)

    async def update_edge(self, source: str, target: str, edge_type: str, weight: float = 1.0, *, channel_id: str | None = None, sentiment_score: float | None = None, event_count_delta: int = 1) -> None:
        await self._db.update_edge(source, target, edge_type, weight, channel_id=channel_id, sentiment_score=sentiment_score, event_count_delta=event_count_delta)

    async def get_edge_weight(self, source: str, target: str, edge_type: str, *, channel_id: str | None = None) -> float:
        return await self._db.get_edge_weight(source, target, edge_type, channel_id=channel_id)

    async def update_social_model(self, *, user_id: str, counterpart_id: str, channel_id: str | None, persona: str | None, affinity: int, trust: float, perception: dict[str, float], reply_latency: float | None = None) -> dict[str, Any]:
        social_model = await self._load_social_model(user_id)
        dimensions = dict(social_model.get("dimensions") or {})
        relationship = await self.get_relationship_stats(user_id, counterpart_id)
        interaction = await self._db.get_edge_summary(user_id, counterpart_id, "interaction", channel_id=channel_id)
        pair_events = int(relationship["a_to_b"]["count"]) + int(relationship["b_to_a"]["count"])
        familiarity_score = self._clamp(pair_events / 12.0)
        trust_score = self._clamp((trust + 10.0) / 20.0)
        correction_score = self._clamp(0.55 + 0.15 * float(perception.get("avoidance", 0.0)) + 0.2 * float(perception.get("manipulation", 0.0)) - 0.15 * trust_score)
        cadence_observed = 0.5 if reply_latency is None else (0.85 if reply_latency <= LATENCY_THRESHOLD else 0.25)
        cadence_score = self._clamp((interaction.get("reciprocity", 0.0) + cadence_observed) / 2.0)
        style_label = (persona or "balanced").strip().lower() or "balanced"
        topic_rows = await self._db.recall_user(user_id, limit=24)
        topic_counts = Counter(topic for topic, _ in topic_rows if isinstance(topic, str) and topic and topic != "social_perception")
        top_topics = [topic for topic, _count in topic_counts.most_common(3)]
        dominant_topic = top_topics[0] if top_topics else "general"
        topic_score = self._clamp(min(1.0, len(top_topics) / 3.0) * 0.7 + familiarity_score * 0.3)

        previous = {
            name: float((dimensions.get(name) or {}).get("score", default))
            for name, default in {
                "familiarity": familiarity_score,
                "trust_rapport": trust_score,
                "topic_affinity": topic_score,
                "cadence_tolerance": cadence_score,
                "correction_sensitivity": correction_score,
            }.items()
        }

        familiarity_score = self._blend(previous["familiarity"], familiarity_score)
        trust_score = self._blend(previous["trust_rapport"], trust_score)
        topic_score = self._blend(previous["topic_affinity"], topic_score)
        cadence_score = self._blend(previous["cadence_tolerance"], cadence_score)
        correction_score = self._blend(previous["correction_sensitivity"], correction_score)

        channel_specific_norms = dict(social_model.get("channel_specific_norms") or {})
        channel_key = str(channel_id) if channel_id else "default"
        channel_specific_norms[channel_key] = {
            "interaction_frequency": int(interaction.get("event_count", 0)),
            "reciprocity": float(interaction.get("reciprocity", 0.0)),
            "sentiment_trend": interaction.get("sentiment_trend", "stable"),
            "preferred_style": style_label,
            "cadence_tolerance": round(cadence_score, 3),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        dimensions.update(
            {
                "familiarity": {"score": round(familiarity_score, 3), "level": self._tier(familiarity_score)},
                "trust_rapport": {"score": round(trust_score, 3), "level": self._tier(trust_score)},
                "preferred_response_style": {"label": style_label, "confidence": round(max(trust_score, familiarity_score), 3)},
                "topic_affinity": {"score": round(topic_score, 3), "level": self._tier(topic_score), "top_topics": top_topics, "dominant_topic": dominant_topic},
                "cadence_tolerance": {"score": round(cadence_score, 3), "level": self._tier(cadence_score)},
                "correction_sensitivity": {"score": round(correction_score, 3), "level": self._tier(correction_score)},
                "channel_specific_norms": {"channels": channel_specific_norms, "active_channel": channel_key},
            }
        )

        social_model = {
            "version": SOCIAL_MODEL_VERSION,
            "dimensions": dimensions,
            "channel_specific_norms": channel_specific_norms,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        await self._save_social_model(user_id, social_model)

        for predicate, object_value, confidence, attributes in (
            ("social_familiarity", dimensions["familiarity"]["level"], familiarity_score, dimensions["familiarity"]),
            ("social_trust_rapport", dimensions["trust_rapport"]["level"], trust_score, dimensions["trust_rapport"]),
            ("social_preferred_response_style", style_label, dimensions["preferred_response_style"]["confidence"], dimensions["preferred_response_style"]),
            ("social_topic_affinity", dominant_topic, topic_score, dimensions["topic_affinity"]),
            ("social_cadence_tolerance", dimensions["cadence_tolerance"]["level"], cadence_score, dimensions["cadence_tolerance"]),
            ("social_correction_sensitivity", dimensions["correction_sensitivity"]["level"], correction_score, dimensions["correction_sensitivity"]),
        ):
            await self._upsert_social_fact(user_id=user_id, predicate=predicate, object_value=str(object_value), confidence=float(confidence), attributes={**attributes, "counterpart_id": counterpart_id, "channel_id": channel_id})
        return social_model

    async def get_social_context_summary(self, source: str, target: str, *, channel_id: str | None = None) -> dict:
        status = await self._db.get_relationship_type(source, target) or "neutral"
        rel = await self.get_relationship_stats(source, target)
        pair_events = int(rel["a_to_b"]["count"]) + int(rel["b_to_a"]["count"])
        familiarity = "high" if pair_events >= 12 else "medium" if pair_events >= 4 else "low"
        interaction = await self._db.get_edge_summary(source, target, "interaction", channel_id=channel_id)
        trust = await self._db.get_trust(source)
        affinity = await self._db.get_affinity(source)
        social_model = await self._load_social_model(source)
        dimensions = dict(social_model.get("dimensions") or {})
        channel_specific_norms = dict(social_model.get("channel_specific_norms") or {})
        active_channel = channel_specific_norms.get(str(channel_id) if channel_id else "default", {})
        evidence = retrieve_user_context(self._graph_memory, str(source), limit=6)
        preferred_style = ((dimensions.get("preferred_response_style") or {}).get("label") or "balanced")
        correction_level = ((dimensions.get("correction_sensitivity") or {}).get("level") or "medium")
        rapport_level = ((dimensions.get("trust_rapport") or {}).get("level") or self._tier((trust + 10.0) / 20.0))
        topic_affinity = dict(dimensions.get("topic_affinity") or {})

        selector_inputs = {
            "social_intent_hints": {
                "preferred_style": preferred_style,
                "clarify_preferred": correction_level == "high",
                "high_rapport_expected": rapport_level == "high",
                "correction_sensitive": correction_level in {"medium", "high"},
                "channel_style": active_channel.get("preferred_style", preferred_style),
                "top_topics": topic_affinity.get("top_topics", []),
            },
            "user_history_affinity": {
                "default": round(self._clamp((affinity + 10.0) / 20.0) * 2.0 - 1.0, 3),
                "intent": round(float((dimensions.get("trust_rapport") or {}).get("score", 0.5)) * 2.0 - 1.0, 3),
                "persona": round(0.25 + 0.5 * float((dimensions.get("familiarity") or {}).get("score", 0.5)), 3),
                "factual": round(0.2 + 0.5 * (1.0 - float((dimensions.get("correction_sensitivity") or {}).get("score", 0.5))), 3),
                "safety": round(0.4 + 0.4 * float((dimensions.get("correction_sensitivity") or {}).get("score", 0.5)), 3),
            },
            "interaction_policy": {
                "response_style": preferred_style,
                "ask_clarifying_on_no_safe": correction_level != "low",
                "style_modifiers": [preferred_style, rapport_level],
                "cadence_tolerance": float((dimensions.get("cadence_tolerance") or {}).get("score", 0.5)),
                "channel_norms": active_channel or {
                    "interaction_frequency": interaction["event_count"],
                    "reciprocity": interaction["reciprocity"],
                    "sentiment_trend": interaction["sentiment_trend"],
                },
            },
        }
        return {
            "relationship_status": status,
            "familiarity_tier": dimensions.get("familiarity", {}).get("level", familiarity),
            "channel_norms": {
                "interaction_frequency": interaction["event_count"],
                "reciprocity": interaction["reciprocity"],
                "sentiment_trend": interaction["sentiment_trend"],
            },
            "interaction_edge": interaction,
            "durable_user_model": {
                "version": SOCIAL_MODEL_VERSION,
                "dimensions": dimensions,
                "relationship_status": status,
                "affinity": affinity,
                "trust": trust,
                "graph_evidence": [item.summary for item in evidence],
            },
            "selector_inputs": selector_inputs,
        }

    async def discover_factions(self, edge_type: str = "ally") -> list[list[str]]:
        try:
            import importlib
            nx = importlib.import_module("networkx")
        except Exception as exc:  # pragma: no cover - defensive
            raise ImportError("networkx is required for faction discovery") from exc
        edges = await self._db.get_edges(edge_type=edge_type)
        graph = nx.Graph()
        for src, tgt, weight in edges:
            if weight > 0:
                graph.add_edge(src, tgt, weight=weight)
        if graph.number_of_nodes() == 0:
            return []
        communities = nx.algorithms.community.greedy_modularity_communities(graph, weight="weight")
        return [sorted(list(c)) for c in communities]

    async def ingest_prism_event(self, event: PrismEvent) -> None:
        await self._db.log_interaction(event.source, event.target, sentiment_score=event.sentiment)
        inferred_targets: set[str] = set()
        if event.target is not None:
            inferred_targets.add(str(event.target))
        if event.referenced_user_id:
            inferred_targets.add(str(event.referenced_user_id))
        for participant in event.thread_participants:
            if participant and participant != event.source:
                inferred_targets.add(participant)
        for user in event.co_occurring_users:
            if user and user != event.source:
                inferred_targets.add(user)
        for target in inferred_targets:
            await self.update_edge(event.source, target, "interaction", 1.0, channel_id=event.channel_id, sentiment_score=event.sentiment)
            await self._update_relationship_type(event.source, target)
        if event.reply_latency is not None:
            delta = 1 if event.reply_latency <= LATENCY_THRESHOLD else -1
            await self._db.adjust_affinity(event.source, delta)
        if event.emoji_counts:
            emoji_target = event.target or event.referenced_user_id
            if emoji_target is not None:
                for emoji, count in event.emoji_counts.items():
                    if emoji in POSITIVE_EMOJIS:
                        await self.update_edge(event.source, emoji_target, "ally", float(count), channel_id=event.channel_id, sentiment_score=event.sentiment)
                    elif emoji in NEGATIVE_EMOJIS:
                        await self.update_edge(event.source, emoji_target, "rival", float(count), channel_id=event.channel_id, sentiment_score=event.sentiment)

    async def close(self) -> None:
        await self._db.close()
