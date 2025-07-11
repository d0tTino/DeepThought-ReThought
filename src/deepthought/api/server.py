from __future__ import annotations

"""FastAPI server exposing memory endpoints via NATS."""

import json
import logging
from collections import OrderedDict
from typing import Dict, List, Optional

import nats
from fastapi import FastAPI, HTTPException
from nats.aio.msg import Msg
from nats.js.client import JetStreamContext
from pydantic import BaseModel

from ..config import get_settings
from ..eda.events import EventSubjects
from ..eda.subscriber import Subscriber
from ..modules.input_handler import InputHandler

logger = logging.getLogger(__name__)

app = FastAPI(title="DeepThought API")


class MemoryCache:
    """Store recent MEMORY_RETRIEVED events."""

    def __init__(self, nats_client: nats.NATS, js_context: JetStreamContext, max_entries: int = 100) -> None:
        self._subscriber = Subscriber(nats_client, js_context)
        self._cache: "OrderedDict[str, Dict]" = OrderedDict()
        self._max = max_entries

    async def _handle_event(self, msg: Msg) -> None:
        input_id = "unknown"
        try:
            data = json.loads(msg.data.decode())
            if not isinstance(data, dict):
                raise ValueError("Invalid payload")
            input_id = data.get("input_id")
            knowledge = data.get("retrieved_knowledge")
            if not isinstance(input_id, str):
                raise ValueError("input_id missing")
            self._cache[input_id] = knowledge
            if len(self._cache) > self._max:
                self._cache.popitem(last=False)
            await msg.ack()
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to process memory event: %s", exc, exc_info=True)
            if hasattr(msg, "nak") and callable(msg.nak):
                try:
                    await msg.nak()
                except Exception:
                    logger.error("Failed to NAK message", exc_info=True)

    def get(self, input_id: str) -> Optional[Dict]:
        return self._cache.get(input_id)

    def search(self, query: str) -> List[Dict]:
        """Return cached entries whose facts contain the query string."""
        results: List[Dict] = []
        q = query.lower()
        for input_id, knowledge in self._cache.items():
            data = knowledge
            if isinstance(data, dict) and "retrieved_knowledge" in data:
                data = data.get("retrieved_knowledge")
            if not isinstance(data, dict):
                continue
            facts = data.get("facts")
            if not isinstance(facts, list):
                continue
            if any(q in str(f).lower() for f in facts):
                results.append({"input_id": input_id, "retrieved_knowledge": knowledge})
        return results

    async def start(self, durable: str = "api_memory_cache") -> bool:
        try:
            await self._subscriber.subscribe(
                subject=EventSubjects.MEMORY_RETRIEVED,
                handler=self._handle_event,
                use_jetstream=True,
                durable=durable,
            )
            return True
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to subscribe to MEMORY_RETRIEVED: %s", exc, exc_info=True)
            return False

    async def stop(self) -> None:
        await self._subscriber.unsubscribe_all()


class MemoryAddRequest(BaseModel):
    text: str


class MemoryQueryRequest(BaseModel):
    query: str


@app.on_event("startup")
async def _startup() -> None:
    settings = get_settings()
    nc = await nats.connect(servers=[settings.nats_url])
    js = nc.jetstream()
    app.state.nc = nc
    app.state.js = js
    app.state.input_handler = InputHandler(nc, js)
    cache = MemoryCache(nc, js)
    await cache.start()
    app.state.memory_cache = cache
    logger.info("API server connected to NATS at %s", settings.nats_url)


@app.on_event("shutdown")
async def _shutdown() -> None:
    cache: MemoryCache = app.state.memory_cache
    await cache.stop()
    nc: nats.NATS = app.state.nc
    if nc.is_connected:
        await nc.drain()


@app.post("/memory/add")
async def add_memory(req: MemoryAddRequest) -> Dict[str, str]:
    try:
        input_id = await app.state.input_handler.process_input(req.text)
        return {"input_id": input_id}
    except Exception as exc:  # pragma: no cover - runtime
        logger.error("Failed to publish INPUT_RECEIVED: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process input")


@app.post("/memory/query")
async def query_memory(req: MemoryQueryRequest) -> Dict[str, List[Dict]]:
    results = app.state.memory_cache.search(req.query)
    return {"results": results}
