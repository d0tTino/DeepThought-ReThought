from __future__ import annotations

import json
import logging
from typing import Optional, Tuple

from nats.aio.client import Client as NATS
from nats.aio.msg import Msg
from nats.js.client import JetStreamContext

from ..eda.events import EventSubjects, MemoryRetrievedPayload
from ..eda.publisher import Publisher
from ..eda.subscriber import Subscriber
from ..ontology import OntologyManager
from .base import BaseService

logger = logging.getLogger(__name__)

Triple = Tuple[str, str, str]
TRIPLE_SUBJECT = "dtr.ontology.triples"


class ReasoningService(BaseService):
    """Service that performs ontology reasoning."""

    def __init__(
        self,
        nats_client: Optional[NATS] = None,
        js_context: Optional[JetStreamContext] = None,
        *,
        ontology: Optional[OntologyManager] = None,
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
        if self._publisher is None:
            self._publisher = Publisher(self._nc, self._js)  # type: ignore[arg-type]
        if self._subscriber is None:
            self._subscriber = Subscriber(self._nc, self._js)  # type: ignore[arg-type]

    async def _handle_triples(self, msg: Msg) -> None:
        input_id = None
        try:
            data = json.loads(msg.data.decode())
            triples = data.get("triples", [])
            input_id = data.get("input_id")
            for t in triples:
                if isinstance(t, (list, tuple)) and len(t) == 3:
                    self._ontology.add_triple(t[0], t[1], t[2])
            facts = self._ontology.infer_facts()
            payload = MemoryRetrievedPayload(
                retrieved_knowledge={"facts": facts, "source": "ontology"},
                input_id=input_id,
            )
            await self._publisher.publish(
                EventSubjects.MEMORY_RETRIEVED,
                payload,
                use_jetstream=True,
                timeout=10.0,
            )
            if hasattr(msg, "ack") and callable(msg.ack):
                await msg.ack()
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to handle triples")

    async def start(self, durable_name: str = "reasoning_service") -> bool:
        self._subscriptions.clear()
        self.add_subscription(
            subject=TRIPLE_SUBJECT,
            handler=self._handle_triples,
            use_jetstream=True,
            durable=durable_name,
        )
        started = await super().start()
        if started:
            logger.info("ReasoningService started")
        return started

    async def stop(self) -> None:
        await super().stop()

    async def __aenter__(self) -> "ReasoningService":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.stop()
