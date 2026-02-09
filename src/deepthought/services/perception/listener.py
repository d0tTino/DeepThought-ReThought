from __future__ import annotations

import inspect
import json
import logging
import os
from typing import Any, Dict, Sequence

from nats.aio.client import Client as NATS
from nats.aio.msg import Msg
from nats.js.client import JetStreamContext

from ...eda.events import (
    EventSubjects,
    InputReceivedPayload,
    PerceptionExtractEvent,
)
from ...eda.subscriber import Subscriber
from .config import PerceptionConfig
from .service import PerceptionService
from .text_utils import hop_aligned_tokens, scrub_tokens

logger = logging.getLogger(__name__)


class PerceptionServiceListener:
    """Subscribe to input events and invoke :class:`PerceptionService`."""

    @staticmethod
    def _infer_attachment_modality(content_type: str | None, filename: str | None) -> str | None:
        ctype = (content_type or "").strip().lower()
        if ctype:
            major = ctype.split("/", 1)[0]
            if major in {"image", "audio", "video"}:
                return major

        name = (filename or "").strip().lower()
        if not name:
            return None

        image_exts = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff"}
        audio_exts = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac", ".opus"}
        video_exts = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}

        for ext_set, modality in ((image_exts, "image"), (audio_exts, "audio"), (video_exts, "video")):
            if any(name.endswith(ext) for ext in ext_set):
                return modality
        return None

    @classmethod
    def _attachment_route(
        cls,
        attachments: Sequence[InputReceivedPayload.AttachmentDescriptor] | None,
    ) -> Dict[str, list[Dict[str, Any]]]:
        routed: Dict[str, list[Dict[str, Any]]] = {"image": [], "audio": [], "video": []}
        for item in attachments or []:
            modality = cls._infer_attachment_modality(item.content_type, item.filename)
            if modality in routed:
                routed[modality].append({
                    "url": item.url,
                    "content_type": item.content_type,
                    "filename": item.filename,
                    "size": item.size,
                })
        return routed

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
        self._config = cfg
        self._enable_asr_transcription = bool(getattr(cfg, "enable_asr_transcription", False))
        hop = getattr(cfg, "text_hop_size", 0.03)
        try:
            hop_value = float(hop)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            hop_value = 0.03
        if hop_value <= 0:  # pragma: no cover - configuration guard
            hop_value = 0.03
        self._text_hop_size = hop_value

    async def start(
        self,
        durable_name: str = "perception_listener",
        *,
        listen_extract: bool = False,
        extract_durable: str | None = None,
    ) -> bool:
        """Begin listening for input events and optional extract requests."""

        ok = await self.start_input(durable_name=durable_name)
        if listen_extract:
            extract_ok = await self.start_extract(
                durable_name=extract_durable or "perception-extract-listener",
            )
            ok = ok and extract_ok
        return ok

    async def start_input(self, durable_name: str = "perception_listener") -> bool:
        """Subscribe to ``dtr.input.received`` events."""

        return await self._subscriber.subscribe(
            subject=EventSubjects.INPUT_RECEIVED,
            handler=self._handle,
            use_jetstream=True,
            durable=durable_name,
        )

    async def start_extract(
        self, durable_name: str = "perception-extract-listener"
    ) -> bool:
        """Subscribe to ``dtr.perception.extract`` events."""

        return await self._subscriber.subscribe(
            subject=EventSubjects.PERCEPTION_EXTRACT,
            handler=self._handle_extract,
            use_jetstream=True,
            durable=durable_name,
        )

    async def handle_input(self, msg: Msg) -> None:
        """Public alias for processing input messages."""

        await self._handle(msg)

    async def handle_extract(self, msg: Msg) -> None:
        """Public alias for processing extraction requests."""

        await self._handle_extract(msg)

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

            payload_user_input: str | None = None
            payload_input_id: str | None = None
            attachments: Sequence[InputReceivedPayload.AttachmentDescriptor] | None = None
            try:
                payload = InputReceivedPayload.from_dict(raw)
            except Exception:
                payload_input_id = raw.get("input_id")
                payload_user_input = raw.get("user_input")
            else:
                payload_input_id = payload.input_id
                payload_user_input = payload.user_input
                attachments = payload.attachments

            routed_attachments = self._attachment_route(attachments)

            message_id = (
                payload_input_id
                or raw.get("message_id")
                or raw.get("input_id")
                or "unknown"
            )

            user_id = (
                raw.get("user_id")
                or (msg.headers.get("user_id") if msg.headers else None)
                or self._default_user_id
            )

            kwargs = {
                "message_id": message_id,
                "input_id": raw.get("input_id") or raw.get("message_id") or payload_input_id,
                "author_id": raw.get("author_id"),
                "channel_id": raw.get("channel_id"),
                "confidence": raw.get("confidence"),
                "user_id": user_id,
                "spans": raw.get("spans"),
                "embeddings": raw.get("embeddings"),
                "encoders": raw.get("encoders"),
                "provenance": raw.get("provenance"),
                "text_tokens": raw.get("text_tokens"),
                "audio_path": raw.get("audio_path"),
                "video_path": raw.get("video_path"),
            }
            if kwargs["audio_path"] is None and routed_attachments["audio"]:
                kwargs["audio_path"] = routed_attachments["audio"][0]["url"]
            if kwargs["video_path"] is None and routed_attachments["video"]:
                kwargs["video_path"] = routed_attachments["video"][0]["url"]
            if isinstance(consent, dict):
                kwargs["audio_opt_in"] = consent.get("audio")
                kwargs["video_opt_in"] = consent.get("video")

            provenance = kwargs.get("provenance")
            if not isinstance(provenance, dict):
                provenance = {}
            attachment_refs = {name: values for name, values in routed_attachments.items() if values}
            if attachment_refs:
                provenance["attachments"] = attachment_refs
                kwargs["provenance"] = provenance

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
            if text_tokens is None and isinstance(payload_user_input, str):
                generated_tokens = hop_aligned_tokens(payload_user_input, self._config.text_hop_size)
                if generated_tokens:
                    kwargs["text_tokens"] = generated_tokens
                    text_tokens = generated_tokens

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
                        sanitized = scrub_tokens(extracted)
                        if sanitized:
                            kwargs["text_tokens"] = sanitized

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

    async def _handle_extract(self, msg: Msg) -> None:
        """Process a perception extraction request."""

        try:
            try:
                raw: Dict[str, Any] = json.loads(msg.data.decode())
            except Exception:
                raw = {}

            event = PerceptionExtractEvent.from_dict(raw)
            payload = event.payload
            if payload is None:
                if hasattr(msg, "ack") and callable(msg.ack):
                    await msg.ack()
                return

            text_tokens = None
            if payload.text_tokens:
                text_tokens = scrub_tokens(payload.text_tokens)
            elif isinstance(payload.text, str) and payload.text.strip():
                hop = payload.text_hop_size or self._text_hop_size
                try:
                    hop_value = float(hop)
                except (TypeError, ValueError):
                    hop_value = self._text_hop_size
                if hop_value <= 0:
                    hop_value = self._text_hop_size
                try:
                    generated = hop_aligned_tokens(payload.text, hop_value)
                except Exception:
                    generated = []
                if generated:
                    text_tokens = generated

            kwargs: Dict[str, Any] = {
                "message_id": payload.message_id,
                "input_id": payload.input_id or payload.message_id,
                "author_id": payload.author_id,
                "channel_id": payload.channel_id,
                "confidence": payload.confidence,
                "user_id": payload.user_id,
                "spans": payload.spans,
                "modality_mask": payload.modality_mask,
                "embeddings": payload.embeddings,
                "encoders": payload.encoders,
                "provenance": payload.provenance,
                "text_tokens": text_tokens,
                "audio_path": payload.audio_path,
                "video_path": payload.video_path,
                "audio_opt_in": payload.audio_opt_in,
                "video_opt_in": payload.video_opt_in,
                "contribution_mask": payload.contribution_mask,
            }
            if payload.retain_media is not None:
                kwargs["retain_media"] = bool(payload.retain_media)

            await self._service.run(**{k: v for k, v in kwargs.items() if v is not None})
            if hasattr(msg, "ack") and callable(msg.ack):
                await msg.ack()
        except Exception:  # pragma: no cover - defensive
            logger.error("Failed to process perception extract request", exc_info=True)
            if hasattr(msg, "nak") and callable(msg.nak):
                try:
                    await msg.nak()
                except Exception:  # pragma: no cover - defensive
                    logger.error("Failed to NAK message", exc_info=True)
            elif hasattr(msg, "ack") and callable(msg.ack):
                try:
                    await msg.ack()
                except Exception:  # pragma: no cover - defensive
                    logger.error("Failed to ack message after extract error", exc_info=True)
