from __future__ import annotations

import json
import logging
from typing import Optional

from nats.aio.client import Client as NATS
from nats.aio.msg import Msg
from nats.js.client import JetStreamContext

from ..perception.social_perception import analyze as analyze_social
from ..eda.events import EventSubjects
from ..eda.publisher import publish_enveloped
from .base import BaseService
from .db_manager import DBManager
from .persona_manager import PersonaManager
from .input_enrichment_service import InputEnrichmentService
from .prism_adapter import PrismAdapter
from .social_graph_memory import SocialGraphMemory

logger = logging.getLogger(__name__)


class SocialGraphService(BaseService):
    """Service that records sentiment and adjusts user affinity."""

    def __init__(
        self,
        nats_client: Optional[NATS] = None,
        js_context: Optional[JetStreamContext] = None,
        db_manager: Optional[DBManager] = None,
        persona_manager: Optional[PersonaManager] = None,
        prism_adapter: Optional[PrismAdapter] = None,
        input_enrichment: InputEnrichmentService | None = None,
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
        self._db = db_manager or DBManager()
        self._memory = SocialGraphMemory(self._db)
        self._persona = persona_manager or PersonaManager(self._db)
        self._prism = prism_adapter or PrismAdapter(self._memory)
        self._input_enrichment = input_enrichment or InputEnrichmentService()

    async def _handle_social_signals_requested(self, msg: Msg) -> None:
        """Respond to SOCIAL_SIGNALS_REQUESTED with context and telemetry events."""
        try:
            data = json.loads(msg.data.decode())
            enriched = self._input_enrichment.parse_input_received_data(data, headers=getattr(msg, "headers", None))
            logger.info("SocialGraphService received input %s", enriched.input_id)
            try:
                perception = analyze_social(enriched.user_input)
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Failed to analyze social perception: %s", exc, exc_info=True)
                perception = {"flirtation": 0.0, "avoidance": 0.0, "manipulation": 0.0}

            resolved_user_id = enriched.resolved_user_id
            await self._db.store_memory(
                resolved_user_id,
                json.dumps(perception),
                topic="social_perception",
            )
            delta = perception.get("flirtation", 0.0) - (
                perception.get("avoidance", 0.0) + perception.get("manipulation", 0.0)
            )
            await self._db.adjust_affinity(resolved_user_id, delta)
            trust_delta = delta - float(perception.get("manipulation", 0.0))
            await self._db.adjust_trust(resolved_user_id, trust_delta)
            affinity = await self._db.get_affinity(resolved_user_id)
            trust = await self._db.get_trust(resolved_user_id)
            persona = await self._persona.get_persona(
                resolved_user_id,
                None,
                channel_id=enriched.channel_id,
            )

            prism_payload = {
                "source": resolved_user_id,
                "target": data.get("target_id") or data.get("reply_to_author_id"),
                "sentiment": float(delta),
                "reply_latency": data.get("reply_latency"),
                "emoji_counts": data.get("emoji_counts") or {},
                "channel_id": enriched.channel_id,
                "referenced_user_id": data.get("referenced_user_id"),
                "thread_participants": data.get("thread_participants") or [],
                "co_occurring_users": data.get("co_occurring_users") or [],
            }
            await self._prism.ingest(prism_payload)

            counterpart_id = str(prism_payload["target"] or prism_payload["referenced_user_id"] or resolved_user_id)
            await self._memory.update_social_model(
                user_id=resolved_user_id,
                counterpart_id=counterpart_id,
                channel_id=enriched.channel_id,
                persona=persona,
                affinity=affinity,
                trust=trust,
                perception=perception,
                reply_latency=prism_payload.get("reply_latency"),
            )
            social_context = await self._memory.get_social_context_summary(
                resolved_user_id,
                counterpart_id,
                channel_id=enriched.channel_id,
            )

            social_snapshot = {
                "input_id": enriched.input_id,
                "user_id": enriched.user_id,
                "author_id": enriched.author_id,
                "channel_id": enriched.channel_id,
                "social_signals": {
                    "perception": perception,
                    "delta": delta,
                    "affinity": affinity,
                    "trust": trust,
                    "persona": persona,
                    "durable_user_model": social_context.get("durable_user_model", {}),
                    "selector_inputs": social_context.get("selector_inputs", {}),
                    "relationship_status": social_context.get("relationship_status", "neutral"),
                    "familiarity_tier": social_context.get("familiarity_tier", "low"),
                    "channel_norms": social_context.get("channel_norms", {}),
                },
            }
            social_update = {
                "input_id": enriched.input_id,
                "user_id": enriched.user_id,
                "author_id": enriched.author_id,
                "channel_id": enriched.channel_id,
                "delta": delta,
                "affinity": affinity,
                "trust": trust,
                "persona": persona,
                "perception": perception,
            }
            await publish_enveloped(
                self._publisher,
                subject=EventSubjects.SOCIAL_UPDATED,
                payload=social_update,
                producer="social_graph_service",
            )
            await publish_enveloped(
                self._publisher,
                subject=EventSubjects.SOCIAL_SIGNALS_RETRIEVED,
                payload=social_snapshot,
                producer="social_graph_service",
            )
            if hasattr(msg, "ack") and callable(msg.ack):
                await msg.ack()
        except (json.JSONDecodeError, ValueError):
            logger.error("Invalid InputReceived payload for social graph", exc_info=True)
            if hasattr(msg, "nak") and callable(msg.nak):
                await msg.nak()
            elif hasattr(msg, "ack") and callable(msg.ack):
                await msg.ack()
        except Exception:  # pragma: no cover - defensive
            logger.error("Failed to handle input", exc_info=True)
            if hasattr(msg, "nak") and callable(msg.nak):
                await msg.nak()
            elif hasattr(msg, "ack") and callable(msg.ack):
                await msg.ack()


    async def _handle_input(self, msg: Msg) -> None:
        """Backward-compatible alias for social signal request handling."""
        await self._handle_social_signals_requested(msg)

    async def start(self, durable_name: str = "social_graph_service") -> bool:
        self._subscriptions.clear()
        self.add_subscription(
            subject=EventSubjects.SOCIAL_SIGNALS_REQUESTED,
            handler=self._handle_social_signals_requested,
            use_jetstream=True,
            durable=durable_name,
        )
        return await super().start()

    async def stop(self) -> None:
        if hasattr(self._db, "close"):
            try:
                await self._db.close()
            except Exception:
                logger.error("Failed to close DB", exc_info=True)
        await super().stop()
