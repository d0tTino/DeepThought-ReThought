from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Mapping

from nats.aio.msg import Msg


@dataclass(frozen=True)
class EnrichedInputPayload:
    """Normalized INPUT_RECEIVED payload values shared by services."""

    input_id: str
    user_input: str
    user_id: str | None
    author_id: str | None
    channel_id: str | None

    @property
    def resolved_user_id(self) -> str:
        return self.author_id or self.user_id or "anonymous"


class InputEnrichmentService:
    """Parse and normalize INPUT_RECEIVED payloads from NATS messages."""

    @staticmethod
    def _normalized_identifier(value: object) -> str | None:
        if not isinstance(value, str):
            return None
        normalized = value.strip()
        return normalized or None

    def parse_input_received(self, msg: Msg) -> EnrichedInputPayload:
        data = json.loads(msg.data.decode())
        if not isinstance(data, dict):
            raise ValueError("InputReceived payload must be a dict")

        input_id = data.get("input_id")
        user_input = data.get("user_input")
        if not isinstance(input_id, str) or not isinstance(user_input, str):
            raise ValueError("Invalid input payload fields")

        headers = getattr(msg, "headers", None)
        if headers is not None and not isinstance(headers, Mapping):
            headers = None

        user_id = self._normalized_identifier(data.get("user_id"))
        if user_id is None and headers:
            user_id = self._normalized_identifier(headers.get("user_id"))

        author_id = self._normalized_identifier(data.get("author_id"))
        if author_id is None and headers:
            author_id = self._normalized_identifier(headers.get("author_id"))
        if author_id is None:
            author_id = user_id

        channel_id = data.get("channel_id")
        if not isinstance(channel_id, str):
            channel_id = None
        if channel_id is None and headers:
            channel_id = headers.get("channel_id")
        if not isinstance(channel_id, str):
            channel_id = None

        return EnrichedInputPayload(
            input_id=input_id,
            user_input=user_input,
            user_id=user_id,
            author_id=author_id,
            channel_id=channel_id,
        )

