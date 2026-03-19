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
from .policy_engine import VersionedPolicyEngine

logger = logging.getLogger(__name__)


class ResponderService:
    """Optional heuristic specialist that emits selector inputs, not the primary bot voice."""

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
        self._policy_engine = VersionedPolicyEngine()

    @staticmethod
    def _bounded_social_features(social_signals: dict) -> dict:
        if not isinstance(social_signals, dict):
            return {}
        channel_norms = social_signals.get("channel_norms")
        if not isinstance(channel_norms, dict):
            channel_norms = {}
        return {
            "relationship_status": social_signals.get("relationship_status", "neutral"),
            "familiarity_tier": social_signals.get("familiarity_tier", "low"),
            "channel_norms": {
                "interaction_frequency": channel_norms.get("interaction_frequency", 0),
                "reciprocity": channel_norms.get("reciprocity", 0.0),
                "sentiment_trend": channel_norms.get("sentiment_trend", "stable"),
            },
        }

    def _compose_prompt_context(self, user_input: str, facts: list[str], social_signals: dict) -> str:
        bounded = self._bounded_social_features(social_signals)
        fact_hint = facts[0] if facts else ""
        return f"user_input={user_input[:160]} | fact={fact_hint[:120]} | social={json.dumps(bounded, sort_keys=True)}"

    def _build_specialist_payload(
        self,
        user_input: str,
        facts: list[str],
        social_signals: dict,
    ) -> tuple[str, float, list[str], str, str]:
        if self._responder_kind in {"factual", "qa", "tool"}:
            fact = facts[0] if facts else "I don't have enough retrieved facts yet"
            return (
                f"Useful grounding fact: {fact}.",
                0.78,
                ["specialist", "factual", "grounded", "tool_ready"],
                "concise",
                "Fact-grounded specialist hint",
            )
        if self._responder_kind in {"persona", "conversational"}:
            prompt_context = self._compose_prompt_context(user_input, facts, social_signals)
            return (
                f"Tone/rapport hint: acknowledge the user warmly while staying aligned with {prompt_context}.",
                0.73,
                ["specialist", "persona", "empathetic", "rapport"],
                "friendly",
                "Persona specialist hint",
            )

        blocked = any(token in user_input.lower() for token in ("harm", "attack", "exploit"))
        if blocked:
            return (
                "Safety hint: refuse harmful assistance and redirect to safer alternatives.",
                0.93,
                ["specialist", "safety", "refusal", "policy"],
                "safety",
                "Safety specialist refusal",
            )
        return (
            "Safety hint: content appears answerable, but avoid escalating risk and keep the reply bounded.",
            0.62,
            ["specialist", "safety", "allow"],
            "safety",
            "Safety specialist allow hint",
        )

    def _build_candidate(
        self,
        user_input: str,
        facts: list[str],
        social_signals: dict,
        policy_artifacts: list[dict],
    ) -> ResponseCandidate:
        source = f"responder:{self._responder_id}"
        text, confidence, tags, style, role_description = self._build_specialist_payload(
            user_input=user_input,
            facts=facts,
            social_signals=social_signals,
        )

        decision = self._policy_engine.evaluate_candidate(
            text=text,
            confidence=confidence,
            prior_artifacts=policy_artifacts,
        )
        safety_passed = decision.allowed
        confidence_components = {"model": confidence, "prior": 0.8}
        calibration = {"slope": 1.0, "bias": 0.0, "version": "heuristic_v1"}

        return ResponseCandidate(
            text=text,
            confidence=confidence,
            source=source,
            safety_passed=safety_passed,
            confidence_components=confidence_components,
            safety_metadata={
                "style": style,
                "policy": "keyword_v1",
                "policy_artifacts": [*policy_artifacts, decision.artifacts],
                "policy_action": decision.action,
                "policy_reason": decision.reason,
                "safety_passed": safety_passed,
            },
            source_metadata={
                "source": source,
                "responder_id": self._responder_id,
                "kind": self._responder_kind,
                "role": "specialist_candidate_producer",
                "role_description": role_description,
                "is_primary_voice": False,
                "calibration": calibration,
                "calibration_metadata": {
                    "calibration": calibration,
                    "confidence_components": confidence_components,
                    "policy_version": self._policy_engine.VERSION,
                },
                "social_features": self._bounded_social_features(social_signals),
                "policy_version": self._policy_engine.VERSION,
            },
            rationale_tags=tags,
        )

    async def _handle_context_event(self, msg: Msg) -> None:
        try:
            data = json.loads(msg.data.decode())
            payload, envelope_meta = decode_payload_or_envelope(EventSubjects.CONTEXT_ASSEMBLED, data)
            user_input = payload.get("user_input") if isinstance(payload.get("user_input"), str) else ""
            facts = payload.get("retrieved_facts") if isinstance(payload.get("retrieved_facts"), list) else []
            social_signals = payload.get("social_signals") if isinstance(payload.get("social_signals"), dict) else {}
            input_id = payload.get("input_id") if isinstance(payload.get("input_id"), str) else "global"
            author_id = payload.get("author_id") if isinstance(payload.get("author_id"), str) else None
            user_id = payload.get("user_id") if isinstance(payload.get("user_id"), str) else None
            channel_id = payload.get("channel_id") if isinstance(payload.get("channel_id"), str) else None

            risk_artifact = self._policy_engine.classify_input_risk(user_input)
            hardening_artifact = self._policy_engine.harden_prompt(user_input, risk_artifact=risk_artifact)
            policy_artifacts = [risk_artifact, hardening_artifact]

            selector_inputs = social_signals.get("selector_inputs") if isinstance(social_signals.get("selector_inputs"), dict) else {}
            out = ResponseCandidatesPayload(
                candidates=[
                    self._build_candidate(
                        user_input=user_input,
                        facts=[str(x) for x in facts],
                        social_signals=social_signals,
                        policy_artifacts=policy_artifacts,
                    )
                ],
                input_id=input_id,
                user_id=author_id or user_id,
                author_id=author_id,
                channel_id=channel_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                interaction_policy=selector_inputs.get("interaction_policy"),
                social_intent_hints=selector_inputs.get("social_intent_hints"),
                user_history_affinity=selector_inputs.get("user_history_affinity"),
            )
            envelope = EventEnvelope.build(
                subject=EventSubjects.RESPONSE_CANDIDATES,
                payload=json.loads(out.to_json()),
                producer=f"ResponderService[{self._responder_id}]",
                trace_id=envelope_meta.get("trace_id") if isinstance(envelope_meta.get("trace_id"), str) else None,
                causation_id=envelope_meta.get("event_id") if isinstance(envelope_meta.get("event_id"), str) else input_id,
            )
            await self._publisher.publish(
                EventSubjects.RESPONSE_CANDIDATES,
                envelope.__dict__,
                use_jetstream=True,
                timeout=10.0,
            )
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
