"""JetStream ledger for motivation signals."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Optional

from nats.aio.client import Client as NATS
from nats.js.client import JetStreamContext


@dataclass
class RewardEvent:
    prompt: str
    response: str
    reward: float
    timestamp: str

    def to_json(self) -> str:
        return json.dumps(asdict(self))


class Ledger:
    """Publish reward events to JetStream."""

    def __init__(self, nats_client: NATS, js_context: JetStreamContext, subject: str = "motivation") -> None:
        self._client = nats_client
        self._js = js_context
        self.subject = subject

    async def publish(self, prompt: str, response: str, reward: float) -> None:
        evt = RewardEvent(prompt=prompt, response=response, reward=reward, timestamp=datetime.utcnow().isoformat())
        await self._js.publish(self.subject, evt.to_json().encode())
