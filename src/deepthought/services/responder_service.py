from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from nats.aio.client import Client as NATS
from nats.aio.msg import Msg
from nats.js.client import JetStreamContext

from ..eda.contracts import EventEnvelope, decode_payload_or_envelope
from ..eda.events import EventSubjects, ResponseCandidate, ResponseCandidatesPayload
from ..eda.publisher import Publisher
from ..eda.subscriber import Subscriber

logger = logging.getLogger(__name__)


class ResponderService:
    """Configurable responder that emits candidate responses for selector ranking."""

    def __init__(
        self,
        nats_client: NATS,
        js_context: JetStreamContext,
        *,
        responder_id: str = "factual",
        responder_kind: str = "factual",
    ) -> None:
        self._publisher = Publisher(nats_client, js_context)
        self._subscriber = Subscriber(nats_client, js_context)
        self._responder_id = responder_id.strip().lower() or "factual"
        self._responder_kind = responder_kind.strip().lower() or self._responder_id

    def _build_candidate(self, user_input: str, facts: list[str]) -> ResponseCandidate:
        source = f"responder:{self._responder_id}"
        if self._responder_kind in {"factual", "qa", "tool"}:
            fact = facts[0] if facts else "I don't have enough retrieved facts yet"
            text = f"Factual answer: {fact}."
            confidence = 0.78
            tags = ["factual", "grounded", "tool_ready"]
            style = "concise"
        elif self._responder_kind in {"persona", "conversational"}:
            text = f"I hear you: {user_input}. I'm here and happy to keep talking."
            confidence = 0.73
            tags = ["persona", "empathetic", "rapport"]
            style = "friendly"
        else:
            blocked = any(token in user_input.lower() for token in ("harm", "attack", "exploit"))
            if blocked:
                text = "I can’t help with harmful actions. I can help with safer alternatives instead."
                confidence = 0.93
                tags = ["safety", "refusal", "policy"]
            else:
                text = "This looks safe to answer. Please continue with what you need."
                confidence = 0.62
                tags = ["safety", "allow"]
            style = "safety"

        return ResponseCandidate(
            text=text,
            confidence=confidence,
            source=source,
            safety_passed=True,
            confidence_components={"model": confidence, "prior": 0.8},
            safety_metadata={"style": style, "policy": "keyword_v1"},
            source_metadata={
                "responder_id": self._responder_id,
                "kind": self._responder_kind,
                "calibration": {"slope": 1.0, "bias": 0.0},
            },
            rationale_tags=tags,
        )

    async def _handle_context_event(self, msg: Msg) -> None:
        try:
            data = json.loads(msg.data.decode())
            payload, envelope_meta = decode_payload_or_envelope(EventSubjects.CONTEXT_ASSEMBLED, data)
            user_input = payload.get("user_input") if isinstance(payload.get("user_input"), str) else ""
            facts = payload.get("retrieved_facts") if isinstance(payload.get("retrieved_facts"), list) else []
            input_id = payload.get("input_id") if isinstance(payload.get("input_id"), str) else "global"
            author_id = payload.get("author_id") if isinstance(payload.get("author_id"), str) else None
            user_id = payload.get("user_id") if isinstance(payload.get("user_id"), str) else None
            channel_id = payload.get("channel_id") if isinstance(payload.get("channel_id"), str) else None

            out = ResponseCandidatesPayload(
                candidates=[self._build_candidate(user_input=user_input, facts=[str(x) for x in facts])],
                input_id=input_id,
                user_id=author_id or user_id,
                author_id=author_id,
                channel_id=channel_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            envelope = EventEnvelope.build(
                subject=EventSubjects.RESPONSE_CANDIDATES,
                payload=json.loads(out.to_json()),
                producer=f"ResponderService[{self._responder_id}]",
                trace_id=envelope_meta.get("trace_id") if isinstance(envelope_meta.get("trace_id"), str) else None,
                causation_id=envelope_meta.get("event_id") if isinstance(envelope_meta.get("event_id"), str) else input_id,
            )
            await self._publisher.publish(EventSubjects.RESPONSE_CANDIDATES, envelope.__dict__, use_jetstream=True, timeout=10.0)
            await msg.ack()
        except Exception:
            logger.error("ResponderService[%s] failed", self._responder_id, exc_info=True)
            if hasattr(msg, "nak") and callable(msg.nak):
                await msg.nak()

    async def start(self, durable_name: str | None = None) -> bool:
        durable = durable_name or f"responder_{self._responder_id}_service"
        await self._subscriber.subscribe(
            subject=EventSubjects.CONTEXT_ASSEMBLED,
            handler=self._handle_context_event,
            use_jetstream=True,
            durable=durable,
        )
        return True

    async def stop(self) -> None:
        await self._subscriber.unsubscribe_all()

    async def __aenter__(self) -> "ResponderService":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.stop()


class FactualResponderService(ResponderService):
    def __init__(self, nats_client: NATS, js_context: JetStreamContext) -> None:
        super().__init__(nats_client, js_context, responder_id="factual", responder_kind="factual")


class PersonaResponderService(ResponderService):
    def __init__(self, nats_client: NATS, js_context: JetStreamContext) -> None:
        super().__init__(nats_client, js_context, responder_id="persona", responder_kind="persona")


class SafetyResponderService(ResponderService):
    def __init__(self, nats_client: NATS, js_context: JetStreamContext) -> None:
        super().__init__(nats_client, js_context, responder_id="safety", responder_kind="safety")
