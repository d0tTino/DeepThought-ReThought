from __future__ import annotations

import json
import logging
from typing import List, Optional, Tuple
from uuid import uuid4

from nats.aio.client import Client as NATS
from nats.aio.msg import Msg
from nats.js.client import JetStreamContext
from rdflib import Namespace

from ..eda.events import (
    EventSubjects,
    InputReceivedPayload,
    ResponseCandidatesPayload,
    ResponseRankedPayload,
    WarningPayload,
)
from ..ontology import OntologyManager
from .base import BaseService

logger = logging.getLogger(__name__)


class ReasoningService(BaseService):
    """Infer new knowledge from LLM responses."""

    def __init__(
        self,
        nats_client: Optional[NATS] = None,
        js_context: Optional[JetStreamContext] = None,
        ontology: OntologyManager | None = None,
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
        self._ontology = ontology or OntologyManager()
        self._ns = Namespace("http://deepthought.local/resp#")

    # Basic parsing of lines into subject predicate object triples
    def _extract_triples(self, text: str) -> List[Tuple[str, str, str]]:
        triples: List[Tuple[str, str, str]] = []
        for line in text.splitlines():
            sent = line.strip().rstrip(".")
            if not sent:
                continue
            parts = sent.split()
            if len(parts) < 3:
                continue
            subj, pred = parts[0], parts[1]
            obj = "_".join(parts[2:])
            triples.append(
                (str(self._ns[subj]), str(self._ns[pred]), str(self._ns[obj]))
            )
        return triples

    async def _analyze_response_text(self, response_text: str) -> None:
        triples = self._extract_triples(response_text)
        if not triples:
            return
        self._ontology.add_triples(triples)
        facts = self._ontology.infer_facts()
        if not facts:
            return
        valid, contradictions = self._ontology.verify_triples(facts)
        if contradictions:
            warn = WarningPayload(message="Contradictory facts", facts=contradictions)
            await self._publisher.publish(
                EventSubjects.WARNING,
                warn,
                use_jetstream=True,
                timeout=10.0,
            )
        if valid:
            text = "; ".join(" ".join(t) for t in valid)
            out = InputReceivedPayload(user_input=text, input_id=str(uuid4()))
            await self._publisher.publish(
                EventSubjects.INPUT_RECEIVED,
                out,
                use_jetstream=True,
                timeout=10.0,
            )

    async def _handle_ranked_response(self, msg: Msg) -> None:
        try:
            data = json.loads(msg.data.decode())
            payload = ResponseRankedPayload.from_dict(data)
            await self._analyze_response_text(payload.final_response)
            if hasattr(msg, "ack") and callable(msg.ack):
                await msg.ack()
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to process RESPONSE_RANKED: %s", exc, exc_info=True)
            if hasattr(msg, "nak") and callable(msg.nak):
                await msg.nak()

    async def _handle_response_candidates(self, msg: Msg) -> None:
        try:
            data = json.loads(msg.data.decode())
            payload = ResponseCandidatesPayload.from_dict(data)
            for candidate in payload.candidates:
                await self._analyze_response_text(candidate.text)
            if hasattr(msg, "ack") and callable(msg.ack):
                await msg.ack()
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to process RESPONSE_CANDIDATES: %s", exc, exc_info=True)
            if hasattr(msg, "nak") and callable(msg.nak):
                await msg.nak()

    async def start(self, durable_name: str = "reasoning_service") -> bool:
        self._subscriptions.clear()
        self.add_subscription(
            subject=EventSubjects.RESPONSE_RANKED,
            handler=self._handle_ranked_response,
            use_jetstream=True,
            durable=f"{durable_name}_ranked",
        )
        self.add_subscription(
            subject=EventSubjects.RESPONSE_CANDIDATES,
            handler=self._handle_response_candidates,
            use_jetstream=True,
            durable=f"{durable_name}_candidates",
        )
        started = await super().start()
        if started:
            logger.info(
                "ReasoningService subscribed to %s and %s",
                EventSubjects.RESPONSE_RANKED,
                EventSubjects.RESPONSE_CANDIDATES,
            )

        return started

    async def stop(self) -> None:
        await super().stop()

    async def __aenter__(self) -> "ReasoningService":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.stop()
