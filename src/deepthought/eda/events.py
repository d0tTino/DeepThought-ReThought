"""
Event definitions for DeepThought reThought.

This module defines the event structures and naming conventions used
in the DeepThought reThought system's event-driven architecture.
"""

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


# Subject naming convention: dtr.<module>.<event_type>
class EventSubjects:
    """
    Defines standard subject names for the DeepThought reThought event system.

    Subject naming convention: dtr.<module>.<event_type>
    """

    # Input events
    INPUT_RECEIVED = "dtr.input.received"

    # Memory events
    MEMORY_RETRIEVED = "dtr.memory.retrieved"

    # LLM events
    RESPONSE_GENERATED = "dtr.llm.response_generated"
    RESPONSE_CANDIDATES = "dtr.response.candidates"
    RESPONSE_RANKED = "dtr.response.ranked"

    # Perception events
    PERCEPTION_EMBEDDINGS = "dtr.perception.embeddings"
    PERCEPTION_EXTRACT = "dtr.perception.extract"

    # Raw chat message events
    CHAT_RAW = "chat.raw"

    # Scheduler events
    REMINDER_TRIGGERED = "dtr.scheduler.reminder_triggered"
    MICRO_TICK = "dtr.scheduler.micro_tick"
    DAILY_STANDUP = "dtr.scheduler.daily_standup"
    WEEKLY_PLANNING = "dtr.scheduler.weekly_planning"

    # Code generation events
    CODE_TEMPLATE_REQUEST = "dtr.codegen.template_request"
    CODE_GENERATED = "dtr.codegen.generated"

    # Planning events
    PLAN_REQUESTED = "dtr.plan.requested"
    PLAN_GENERATED = "dtr.plan.generated"

    # BDI agent events
    BDI_INTENTION = "dtr.bdi.intention"

    # Warning events
    WARNING = "dtr.warning"

    # Other potential event subjects can be added here as the system expands
    # e.g., ERROR = "dtr.error"
    # e.g., METRICS = "dtr.metrics.reported"


@dataclass
class EventPayload:
    """Base class for all event payloads in the system."""

    def to_json(self) -> str:
        """Convert the payload to a JSON string."""
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, json_str: str) -> "EventPayload":
        """Create a payload instance from a JSON string."""
        data = json.loads(json_str)
        return cls(**data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EventPayload":
        """Create a payload instance from a dictionary."""
        return cls(**data)


@dataclass
class InputReceivedPayload(EventPayload):
    """Payload for input received events."""

    user_input: str
    input_id: Optional[str] = None
    timestamp: Optional[str] = None
    consent: Optional[bool] = None
    message_id: Optional[str] = None
    channel_id: Optional[str] = None
    guild_id: Optional[str] = None
    author_id: Optional[str] = None
    author_name: Optional[str] = None
    author_is_bot: Optional[bool] = None


@dataclass
class MemoryRetrievedPayload(EventPayload):
    """Payload for memory retrieved events."""

    retrieved_knowledge: Dict[str, Any]
    user_input: Optional[str] = None
    input_id: Optional[str] = None
    user_id: Optional[str] = None
    channel_id: Optional[str] = None
    author_id: Optional[str] = None
    author_name: Optional[str] = None
    channel_context: Optional[str] = None
    recent_turn_summary: Optional[str] = None
    timestamp: Optional[str] = None


@dataclass
class ResponseGeneratedPayload(EventPayload):
    """Payload for LLM response generated events."""

    final_response: str
    input_id: Optional[str] = None
    user_id: Optional[str] = None
    timestamp: Optional[str] = None
    confidence: Optional[float] = None


@dataclass
class ResponseCandidate(EventPayload):
    """A single candidate response produced by a responder."""

    text: str
    confidence: float = 0.0
    source: Optional[str] = None
    safety_passed: Optional[bool] = None


@dataclass
class ResponseCandidatesPayload(EventPayload):
    """Payload for candidate response arrays produced by responders."""

    candidates: list[ResponseCandidate]
    input_id: Optional[str] = None
    user_id: Optional[str] = None
    channel_id: Optional[str] = None
    author_id: Optional[str] = None
    timestamp: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResponseCandidatesPayload":
        raw_candidates = data.get("candidates") or []
        candidates = [
            candidate
            if isinstance(candidate, ResponseCandidate)
            else ResponseCandidate(**candidate)
            for candidate in raw_candidates
            if isinstance(candidate, (dict, ResponseCandidate))
        ]
        return cls(
            candidates=candidates,
            input_id=data.get("input_id"),
            user_id=data.get("user_id"),
            channel_id=data.get("channel_id"),
            author_id=data.get("author_id"),
            timestamp=data.get("timestamp"),
        )


@dataclass
class ResponseRankedPayload(EventPayload):
    """Payload for final response chosen from candidate responses."""

    final_response: str
    input_id: Optional[str] = None
    user_id: Optional[str] = None
    channel_id: Optional[str] = None
    author_id: Optional[str] = None
    timestamp: Optional[str] = None
    confidence: Optional[float] = None
    source: Optional[str] = None
    candidates: list[ResponseCandidate] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResponseRankedPayload":
        raw_candidates = data.get("candidates") or []
        candidates = [
            candidate
            if isinstance(candidate, ResponseCandidate)
            else ResponseCandidate(**candidate)
            for candidate in raw_candidates
            if isinstance(candidate, (dict, ResponseCandidate))
        ]
        return cls(
            final_response=data["final_response"],
            input_id=data.get("input_id"),
            user_id=data.get("user_id"),
            channel_id=data.get("channel_id"),
            author_id=data.get("author_id"),
            timestamp=data.get("timestamp"),
            confidence=data.get("confidence"),
            source=data.get("source"),
            candidates=candidates,
        )


@dataclass
class ReminderTriggeredPayload(EventPayload):
    """Payload for scheduled reminder events."""

    message: str
    reminder_id: Optional[str] = None
    timestamp: Optional[str] = None


@dataclass
class TickPayload(EventPayload):
    """Payload for periodic scheduler ticks."""

    timestamp: str


@dataclass
class CodeTemplatePayload(EventPayload):
    """Payload for requesting code generation from a template."""

    template: str
    variables: Dict[str, Any]
    input_id: Optional[str] = None
    timestamp: Optional[str] = None


@dataclass
class CodeGeneratedPayload(EventPayload):
    """Payload for returning generated code and result."""

    code: str
    result: str
    input_id: Optional[str] = None
    timestamp: Optional[str] = None


@dataclass
class PlanRequestedPayload(EventPayload):
    """Payload requesting a plan for a goal."""

    goal: str
    input_id: Optional[str] = None
    timestamp: Optional[str] = None


@dataclass
class PlanGeneratedPayload(EventPayload):
    """Payload containing a generated plan."""

    plan: list[str]
    input_id: Optional[str] = None
    timestamp: Optional[str] = None


@dataclass
class EncoderMetadata(EventPayload):
    """Metadata describing an encoder used for perception."""

    name: str
    modality: Optional[str] = None
    dim: Optional[int] = None
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModalityEmbeddings(EventPayload):
    """Embeddings and spans associated with a single modality."""

    # Each span is represented as [start_ms, end_ms]
    spans: list[list[int]] = field(default_factory=list)
    embeddings: list[list[float]] = field(default_factory=list)
    encoders: list[EncoderMetadata] = field(default_factory=list)
    mask: list[bool] | None = None


@dataclass
class PerceptionEmbeddingsPayload(EventPayload):
    """Payload containing embeddings produced by perception."""

    message_id: str
    user_id: str
    fused: Optional[list[list[float]]] = None
    spans: list[list[int]] = field(default_factory=list)
    modality_mask: Dict[str, list[bool]] = field(default_factory=dict)
    by_modality: Dict[str, ModalityEmbeddings] = field(default_factory=dict)
    contribution_mask: Dict[str, list[bool]] = field(default_factory=dict)


    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PerceptionEmbeddingsPayload":
        raw_by_modality = data.get("by_modality") or {}
        by_modality: Dict[str, ModalityEmbeddings] = {}
        if isinstance(raw_by_modality, dict):
            for name, meta in raw_by_modality.items():
                if isinstance(meta, ModalityEmbeddings):
                    meta_dict = asdict(meta)
                else:
                    meta_dict = dict(meta) if isinstance(meta, dict) else {}

                mask = meta_dict.get("mask")
                mask_list = [bool(value) for value in mask] if mask is not None else None
                spans = [
                    [int(span[0]), int(span[1])]
                    for span in meta_dict.get("spans", [])
                    if isinstance(span, (list, tuple)) and len(span) >= 2
                ]
                embeddings = [
                    [float(x) for x in emb]
                    for emb in meta_dict.get("embeddings", [])
                    if isinstance(emb, (list, tuple))
                ]
                encoders = [
                    enc if isinstance(enc, EncoderMetadata) else EncoderMetadata(**enc)
                    for enc in meta_dict.get("encoders", [])
                ]
                by_modality[name] = ModalityEmbeddings(
                    spans=spans,
                    embeddings=embeddings,
                    encoders=encoders,
                    mask=mask_list,
                )

        fused_raw = data.get("fused")
        fused_vectors: Optional[list[list[float]]] = None
        if fused_raw is not None:
            fused_vectors = []
            if isinstance(fused_raw, (list, tuple)):
                if fused_raw and isinstance(fused_raw[0], (int, float)):
                    fused_vectors.append([float(x) for x in fused_raw])
                else:
                    for vector in fused_raw:
                        if isinstance(vector, (list, tuple)):
                            fused_vectors.append([float(x) for x in vector])
            if not fused_vectors:
                fused_vectors = None

        spans = [
            [int(span[0]), int(span[1])]
            for span in data.get("spans", [])
            if isinstance(span, (list, tuple)) and len(span) >= 2
        ]

        modality_mask = {
            name: [bool(flag) for flag in mask]
            for name, mask in (data.get("modality_mask") or {}).items()
        }

        contribution_mask = {
            name: [bool(flag) for flag in mask]
            for name, mask in (data.get("contribution_mask") or {}).items()
        }

        return cls(
            message_id=data["message_id"],
            user_id=data["user_id"],
            fused=fused_vectors,
            spans=spans,
            modality_mask=modality_mask,
            by_modality=by_modality,
            contribution_mask=contribution_mask,
        )

    @classmethod
    def from_json(cls, json_str: str) -> "PerceptionEmbeddingsPayload":
        return cls.from_dict(json.loads(json_str))


@dataclass
class PerceptionEmbeddingsEvent(EventPayload):
    """Event wrapping perception embeddings with top-level metadata."""

    event: str = EventSubjects.PERCEPTION_EMBEDDINGS
    version: int = 1
    encoders: list[EncoderMetadata] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)
    payload: PerceptionEmbeddingsPayload | None = None

    def to_json(self) -> str:
        """Return a JSON representation with flattened payload fields."""

        payload_dict = asdict(self.payload) if self.payload else {}
        base = {
            "event": self.event,
            "version": self.version,
            "encoders": [asdict(enc) for enc in self.encoders],
            "provenance": dict(self.provenance),
        }
        combined = {**base, **payload_dict}
        if self.payload:
            combined["payload"] = payload_dict
        return json.dumps(combined)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PerceptionEmbeddingsEvent":
        # Deduplicate encoders by (name, modality) to mirror publisher behaviour
        enc_map: dict[tuple[str, str | None], EncoderMetadata] = {}
        for enc in data.get("encoders", []):
            key = (enc.get("name"), enc.get("modality"))
            enc_map.setdefault(key, EncoderMetadata(**enc))
        encoders = list(enc_map.values())
        payload_data = data.get("payload")
        if not payload_data:
            payload_keys = {
                "message_id",
                "user_id",
                "fused",
                "spans",
                "modality_mask",
                "by_modality",
                "contribution_mask",

            }
            payload_data = {k: data[k] for k in payload_keys if k in data}
        payload = PerceptionEmbeddingsPayload.from_dict(payload_data) if payload_data else None
        return cls(
            event=data.get("event", EventSubjects.PERCEPTION_EMBEDDINGS),
            version=data.get("version", 1),
            encoders=encoders,
            provenance=data.get("provenance", {}),
            payload=payload,
        )

    @classmethod
    def from_json(cls, json_str: str) -> "PerceptionEmbeddingsEvent":
        return cls.from_dict(json.loads(json_str))


@dataclass
class PerceptionExtractPayload(EventPayload):
    """Payload describing a perception extraction request."""

    message_id: str
    user_id: str
    text: Optional[str] = None
    text_tokens: Optional[list[list[Any]]] = None
    embeddings: Optional[list[list[float]]] = None
    spans: Optional[list[list[int]]] = None
    modality_mask: Dict[str, list[bool]] = field(default_factory=dict)
    contribution_mask: Dict[str, list[bool]] = field(default_factory=dict)
    encoders: list[Dict[str, Any]] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)
    audio_path: Optional[str] = None
    video_path: Optional[str] = None
    audio_opt_in: Optional[bool] = None
    video_opt_in: Optional[bool] = None
    retain_media: Optional[bool] = None
    text_hop_size: Optional[float] = None

    @staticmethod
    def _parse_tokens(raw_tokens: Any) -> Optional[list[list[Any]]]:
        if not raw_tokens:
            return None
        tokens: list[list[Any]] = []
        if isinstance(raw_tokens, (list, tuple)):
            for token in raw_tokens:
                if not isinstance(token, (list, tuple)) or len(token) < 3:
                    continue
                word = str(token[0])
                try:
                    start = float(token[1])
                    end = float(token[2])
                except (TypeError, ValueError):
                    continue
                tokens.append([word, start, end])
        return tokens or None

    @staticmethod
    def _parse_spans(raw_spans: Any) -> Optional[list[list[int]]]:
        if raw_spans is None:
            return None
        spans: list[list[int]] = []
        for span in raw_spans:
            if not isinstance(span, (list, tuple)) or len(span) < 2:
                continue
            try:
                start = int(span[0])
                end = int(span[1])
            except (TypeError, ValueError):
                continue
            spans.append([start, end])
        return spans or None

    @staticmethod
    def _parse_embeddings(raw_embeddings: Any) -> Optional[list[list[float]]]:
        if raw_embeddings is None:
            return None
        embeddings: list[list[float]] = []
        if isinstance(raw_embeddings, (list, tuple)):
            if raw_embeddings and isinstance(raw_embeddings[0], (int, float)):
                embeddings.append([float(x) for x in raw_embeddings])
            else:
                for vector in raw_embeddings:
                    if isinstance(vector, (list, tuple)):
                        embeddings.append([float(x) for x in vector])
        return embeddings or None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PerceptionExtractPayload":
        tokens = cls._parse_tokens(
            data.get("text_tokens") or data.get("tokens")
        )
        text = data.get("text")
        if text is None and isinstance(data.get("user_input"), str):
            text = data["user_input"]

        modality_mask: Dict[str, list[bool]] = {}
        for name, flags in (data.get("modality_mask") or {}).items():
            modality_mask[name] = [bool(flag) for flag in flags]

        contribution_mask: Dict[str, list[bool]] = {}
        for name, flags in (data.get("contribution_mask") or {}).items():
            contribution_mask[name] = [bool(flag) for flag in flags]

        encoders_raw = data.get("encoders") or []
        encoders: list[Dict[str, Any]] = []
        for enc in encoders_raw:
            if isinstance(enc, dict):
                encoders.append(dict(enc))

        provenance = data.get("provenance")
        if provenance is None or not isinstance(provenance, dict):
            provenance = {}

        consent = data.get("consent")
        audio_opt_in = data.get("audio_opt_in")
        video_opt_in = data.get("video_opt_in")
        if isinstance(consent, dict):
            audio_opt_in = consent.get("audio") if audio_opt_in is None else audio_opt_in
            video_opt_in = consent.get("video") if video_opt_in is None else video_opt_in

        return cls(
            message_id=data["message_id"],
            user_id=data["user_id"],
            text=text,
            text_tokens=tokens,
            embeddings=cls._parse_embeddings(data.get("embeddings") or data.get("fused")),
            spans=cls._parse_spans(data.get("spans")),
            modality_mask=modality_mask,
            contribution_mask=contribution_mask,
            encoders=encoders,
            provenance=provenance,
            audio_path=data.get("audio_path"),
            video_path=data.get("video_path"),
            audio_opt_in=audio_opt_in,
            video_opt_in=video_opt_in,
            retain_media=data.get("retain_media"),
            text_hop_size=data.get("text_hop_size") or data.get("tokens_hop_size"),
        )


@dataclass
class PerceptionExtractEvent(EventPayload):
    """Event describing a perception extraction request."""

    event: str = EventSubjects.PERCEPTION_EXTRACT
    version: int = 1
    payload: Optional[PerceptionExtractPayload] = None

    def to_json(self) -> str:
        base = {
            "event": self.event,
            "version": self.version,
        }
        payload_dict = asdict(self.payload) if self.payload else {}
        if self.payload:
            base["payload"] = payload_dict
        else:
            base.update(payload_dict)
        return json.dumps({**base, **payload_dict})

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PerceptionExtractEvent":
        payload_data = data.get("payload")
        if not payload_data:
            payload_keys = {
                "message_id",
                "user_id",
                "text",
                "text_tokens",
                "tokens",
                "embeddings",
                "fused",
                "spans",
                "modality_mask",
                "contribution_mask",
                "encoders",
                "provenance",
                "audio_path",
                "video_path",
                "audio_opt_in",
                "video_opt_in",
                "retain_media",
                "text_hop_size",
                "tokens_hop_size",
            }
            payload_data = {k: data[k] for k in payload_keys if k in data}
        payload = (
            PerceptionExtractPayload.from_dict(payload_data)
            if payload_data
            else None
        )
        return cls(
            event=data.get("event", EventSubjects.PERCEPTION_EXTRACT),
            version=data.get("version", 1),
            payload=payload,
        )

    @classmethod
    def from_json(cls, json_str: str) -> "PerceptionExtractEvent":
        return cls.from_dict(json.loads(json_str))


@dataclass
class BDIIntentionPayload(EventPayload):
    """Payload representing a BDI intention."""

    goal: str
    priority: int


@dataclass
class WarningPayload(EventPayload):
    """Payload for ontology or system warnings."""

    message: str
    facts: Optional[list[tuple[str, str, str]]] = None
