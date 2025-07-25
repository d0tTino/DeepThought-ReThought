from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

from nats.aio.client import Client as NATS
from nats.aio.msg import Msg
from nats.js.client import JetStreamContext

from ..eda.events import (
    EventSubjects,
    MemoryRetrievedPayload,
    PlanGeneratedPayload,
    PlanRequestedPayload,
)
from ..planning import L2PTranslator, plan
from .base import BaseService

logger = logging.getLogger(__name__)


class PlanningService(BaseService):
    """Simple Belief-Desire-Intention loop."""

    def __init__(
        self,
        nats_client: Optional[NATS] = None,
        js_context: Optional[JetStreamContext] = None,
        desires_file: str = "desires.json",
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
        self._desires_file = desires_file
        self._translator = L2PTranslator()
        self._desires: List[str] = []
        self._beliefs: List[str] = []
        self._load_desires()

    def _load_desires(self) -> None:
        try:
            text = Path(self._desires_file).read_text(encoding="utf-8")
            data = json.loads(text)
            ds = data.get("desires", [])
            if isinstance(ds, list):
                self._desires = [str(d) for d in ds]
            logger.info(
                "Loaded %d desires from %s", len(self._desires), self._desires_file
            )
        except FileNotFoundError:
            logger.warning("Desires file %s not found", self._desires_file)
        except Exception:  # pragma: no cover - defensive
            logger.error(
                "Failed to load desires file %s", self._desires_file, exc_info=True
            )

    async def _handle_memory(self, msg: Msg) -> None:
        try:
            data = json.loads(msg.data.decode())
            payload = MemoryRetrievedPayload.from_dict(data)
            self._beliefs = payload.retrieved_knowledge.get("facts", [])
            await self._form_intentions(payload.input_id)
            if hasattr(msg, "ack") and callable(msg.ack):
                await msg.ack()
        except Exception:  # pragma: no cover - defensive
            logger.error("Failed to process memory event", exc_info=True)
            if hasattr(msg, "nak") and callable(msg.nak):
                await msg.nak()

    async def _handle_plan_request(self, msg: Msg) -> None:
        try:
            data = json.loads(msg.data.decode())
            payload = PlanRequestedPayload.from_dict(data)
            domain, problem = self._translator.translate(payload.goal)
            actions = plan(domain, problem)
            out = PlanGeneratedPayload(plan=actions, input_id=payload.input_id)
            await self._publisher.publish(
                EventSubjects.PLAN_GENERATED,
                out,
                use_jetstream=True,
                timeout=10.0,
            )
            if hasattr(msg, "ack") and callable(msg.ack):
                await msg.ack()
        except Exception:  # pragma: no cover - defensive
            logger.error("Failed to generate plan", exc_info=True)
            if hasattr(msg, "nak") and callable(msg.nak):
                await msg.nak()

    async def _handle_plan(self, msg: Msg) -> None:
        try:
            data = json.loads(msg.data.decode())
            payload = PlanGeneratedPayload.from_dict(data)
            for action in payload.plan:
                await self._publisher.publish(
                    EventSubjects.CHAT_RAW,
                    action,
                    use_jetstream=True,
                    timeout=10.0,
                )
            if hasattr(msg, "ack") and callable(msg.ack):
                await msg.ack()
        except Exception:  # pragma: no cover - defensive
            logger.error("Failed to process plan", exc_info=True)
            if hasattr(msg, "nak") and callable(msg.nak):
                await msg.nak()

    async def _form_intentions(self, input_id: str | None) -> None:
        for goal in self._desires:
            payload = PlanRequestedPayload(goal=goal, input_id=input_id)
            await self._publisher.publish(
                EventSubjects.PLAN_REQUESTED,
                payload,
                use_jetstream=True,
                timeout=10.0,
            )

    async def start(self, durable_name: str = "planning_service") -> bool:
        self._subscriptions.clear()
        self.add_subscription(
            subject=EventSubjects.MEMORY_RETRIEVED,
            handler=self._handle_memory,
            use_jetstream=True,
            durable=f"{durable_name}_belief",
        )
        self.add_subscription(
            subject=EventSubjects.PLAN_GENERATED,
            handler=self._handle_plan,
            use_jetstream=True,
            durable=f"{durable_name}_plan",
        )
        self.add_subscription(
            subject=EventSubjects.PLAN_REQUESTED,
            handler=self._handle_plan_request,
            use_jetstream=True,
            durable=f"{durable_name}_request",
        )
        started = await super().start()
        if started:
            logger.info("PlanningService started")
        return started

    async def stop(self) -> None:
        await super().stop()

    async def __aenter__(self) -> "PlanningService":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.stop()
