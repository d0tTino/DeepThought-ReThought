"""Quest log service backed by JetStream."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict

from ...bus import Publisher, Subscriber
from ...eda.events import EventSubjects
from ..base import BaseService

logger = logging.getLogger(__name__)


class QuestLogService(BaseService):
    """Track quest lifecycle events and emit snapshots and completions."""

    CREATE_DURABLE = "ql_create_v1"
    UPDATE_DURABLE = "ql_update_v1"

    def __init__(self, subscriber: Subscriber, publisher: Publisher) -> None:
        super().__init__(subscriber, publisher)
        self._quests: Dict[str, Dict[str, Any]] = {}
        self._start_lock = asyncio.Lock()
        self._started = False
        self._service_id = "questlog_service"

    async def start(self) -> None:
        if self._started:
            return
        async with self._start_lock:
            if self._started:
                return
            await self._subscriber.subscribe(
                subject=EventSubjects.QUEST_CREATE,
                handler=self._handle_create,
                use_jetstream=True,
                durable=self.CREATE_DURABLE,
            )
            await self._subscriber.subscribe(
                subject=EventSubjects.QUEST_UPDATE,
                handler=self._handle_update,
                use_jetstream=True,
                durable=self.UPDATE_DURABLE,
            )
            self._started = True
            logger.info("QuestLogService listening on %s and %s", EventSubjects.QUEST_CREATE, EventSubjects.QUEST_UPDATE)

    async def _handle_create(self, msg: Any) -> None:
        payload = self._decode_message(msg)
        quest_id = payload.get("quest_id")
        if not quest_id:
            logger.warning("Quest create event missing quest_id: %s", payload)
            await self._ack_message(msg)
            return

        meta = payload.get("meta", {})
        if meta.get("source") == self._service_id:
            await self._ack_message(msg)
            return

        quest_state = {
            "quest_id": quest_id,
            "name": payload.get("name"),
            "description": payload.get("description"),
            "status": payload.get("status", "created"),
            "progress": payload.get("progress", 0.0),
            "metadata": payload.get("metadata", {}),
        }
        self._quests[quest_id] = quest_state

        snapshot = {
            "quests": list(self._quests.values()),
            "changed": quest_state,
            "meta": {"source": self._service_id},
        }
        await self._publish(EventSubjects.QUEST_SNAPSHOT, snapshot)
        await self._ack_message(msg)

    async def _handle_update(self, msg: Any) -> None:
        payload = self._decode_message(msg)
        quest_id = payload.get("quest_id")
        if not quest_id:
            logger.warning("Quest update event missing quest_id: %s", payload)
            await self._ack_message(msg)
            return

        meta = payload.get("meta", {})
        if meta.get("source") == self._service_id:
            await self._ack_message(msg)
            return

        quest_state = self._quests.setdefault(
            quest_id,
            {
                "quest_id": quest_id,
                "name": payload.get("name"),
                "description": payload.get("description"),
                "status": "created",
                "progress": 0.0,
                "metadata": {},
            },
        )
        if "status" in payload:
            quest_state["status"] = payload["status"]
        if "progress" in payload:
            quest_state["progress"] = payload["progress"]
        if "metadata" in payload and isinstance(payload["metadata"], dict):
            quest_state.setdefault("metadata", {}).update(payload["metadata"])

        snapshot = {
            "quests": list(self._quests.values()),
            "changed": quest_state,
            "meta": {"source": self._service_id},
        }
        await self._publish(EventSubjects.QUEST_SNAPSHOT, snapshot)

        status = quest_state.get("status", "").lower()
        if status in {"completed", "done", "finished"}:
            done_payload = {
                "quest_id": quest_id,
                "result": {
                    "status": status,
                    "progress": quest_state.get("progress", 1.0),
                },
                "meta": {"source": self._service_id},
            }
            await self._publish(EventSubjects.QUEST_DONE, done_payload)

        await self._ack_message(msg)
