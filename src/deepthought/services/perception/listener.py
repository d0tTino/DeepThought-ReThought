from __future__ import annotations

import inspect
import json
import logging
import os
from typing import Any, Dict, Sequence

from nats.aio.client import Client as NATS
from nats.aio.msg import Msg
from nats.js.client import JetStreamContext

from ...eda.events import EventSubjects, InputReceivedPayload
from ...eda.subscriber import Subscriber
from .config import PerceptionConfig
from .service import PerceptionService

logger = logging.getLogger(__name__)


class PerceptionServiceListener:
    """Subscribe to input events and invoke :class:`PerceptionService`."""

    def __init__(
        self,
        service: PerceptionService,
        nats_client: NATS,
        js_context: JetStreamContext,
        *,
        default_user_id: str = "user",
        asr: Any | None = None,
    ) -> None:
        self._service = service
        self._subscriber = Subscriber(nats_client, js_context)
        self._default_user_id = default_user_id
        self._asr = asr
        cfg = PerceptionConfig()
        self._enable_asr_transcription = bool(getattr(cfg, "enable_asr_transcription", False))
        hop = getattr(cfg, "text_hop_size", 0.03)
        try:
            hop_value = float(hop)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            hop_value = 0.03
        if hop_value <= 0:  # pragma: no cover - configuration guard
            hop_value = 0.03
        self._text_hop_size = hop_value

    async def start(self, durable_name: str = "perception_listener") -> bool:
        """Begin listening for input events."""
        return await self._subscriber.subscribe(
            subject=EventSubjects.INPUT_RECEIVED,
            handler=self._handle,
            use_jetstream=True,
            durable=durable_name,
        )

    async def _handle(self, msg: Msg) -> None:
        """Decode event payload and dispatch to the service."""
        try:
            try:
                raw: Dict[str, Any] = json.loads(msg.data.decode())
            except Exception:
                raw = {}

            consent = raw.get("consent")
            if os.getenv("PERCEPTION_REQUIRE_CONSENT", "").lower() in {"1", "true", "yes"}:
                granted = bool(consent)
                if isinstance(consent, dict):
                    granted = consent.get("general") is True
                if not granted:
                    if hasattr(msg, "ack") and callable(msg.ack):
                        await msg.ack()
                    return

            try:
                payload = InputReceivedPayload.from_dict(raw)
                message_id = payload.input_id or "unknown"
            except Exception:
                message_id = raw.get("message_id") or "unknown"

            user_id = (
                raw.get("user_id")
                or (msg.headers.get("user_id") if msg.headers else None)
                or self._default_user_id
            )

            kwargs = {
                "message_id": message_id,
                "user_id": user_id,
                "spans": raw.get("spans"),
                "embeddings": raw.get("embeddings"),
                "encoders": raw.get("encoders"),
                "provenance": raw.get("provenance"),
                "text_tokens": raw.get("text_tokens"),
                "audio_path": raw.get("audio_path"),
                "video_path": raw.get("video_path"),
            }
            if isinstance(consent, dict):
                kwargs["audio_opt_in"] = consent.get("audio")
                kwargs["video_opt_in"] = consent.get("video")

            if (
                kwargs.get("text_tokens") is None
                and isinstance(raw.get("user_input"), str)
            ):
                text = raw["user_input"].strip()
                if text:
                    hop = self._text_hop_size or 0.03
                    tokens = []
                    start = 0.0
                    words = text.split()
                    if not words:
                        words = [text]
                    for word in words:
                        end = start + hop
                        tokens.append((word, start, end))
                        start = end
                    kwargs["text_tokens"] = tokens

            audio_path = kwargs.get("audio_path")
            text_tokens = kwargs.get("text_tokens")
            audio_opt_in = kwargs.get("audio_opt_in")
            if (
                text_tokens is None
                and audio_path
                and self._asr is not None
                and self._enable_asr_transcription
                and audio_opt_in is True
            ):
                try:
                    tokens = self._asr.transcribe(audio_path)
                    if inspect.isawaitable(tokens):
                        tokens = await tokens
                except Exception:
                    logger.error("ASR transcription failed for message %s", message_id, exc_info=True)
                else:
                    extracted: Sequence[Any] | None = None
                    if isinstance(tokens, dict):
                        extracted = tokens.get("tokens") or tokens.get("text_tokens")
                    elif hasattr(tokens, "tokens"):
                        extracted = getattr(tokens, "tokens")
                    elif isinstance(tokens, Sequence) and not isinstance(tokens, (str, bytes, bytearray)):
                        extracted = tokens

                    if extracted is not None:
                        kwargs["text_tokens"] = extracted

            await self._service.run(**{k: v for k, v in kwargs.items() if v is not None})
            if hasattr(msg, "ack") and callable(msg.ack):
                await msg.ack()
        except Exception:  # pragma: no cover - defensive
            logger.error("Failed to process perception input", exc_info=True)
            if hasattr(msg, "nak") and callable(msg.nak):
                try:
                    await msg.nak()
                except Exception:  # pragma: no cover - defensive
                    logger.error("Failed to NAK message", exc_info=True)
            elif hasattr(msg, "ack") and callable(msg.ack):
                try:
                    await msg.ack()
                except Exception:  # pragma: no cover - defensive
                    logger.error("Failed to ack message after error", exc_info=True)
