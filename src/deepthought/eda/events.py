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

    # Perception events
    PERCEPTION_EMBEDDINGS = "dtr.perception.embeddings"

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


@dataclass
class MemoryRetrievedPayload(EventPayload):
    """Payload for memory retrieved events."""

    retrieved_knowledge: Dict[str, Any]
    input_id: Optional[str] = None
    timestamp: Optional[str] = None


@dataclass
class ResponseGeneratedPayload(EventPayload):
    """Payload for LLM response generated events."""

    final_response: str
    input_id: Optional[str] = None
    timestamp: Optional[str] = None
    confidence: Optional[float] = None


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


@dataclass
class PerceptionEmbeddingsPayload(EventPayload):
    """Payload containing embeddings produced by perception."""

    message_id: str
    user_id: str
    fused: Optional[list[list[float]]] = None
    by_modality: Dict[str, ModalityEmbeddings] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PerceptionEmbeddingsPayload":
        by_modality = {
            name: ModalityEmbeddings(
                spans=[[int(span[0]), int(span[1])] for span in meta.get("spans", [])],
                embeddings=[list(map(float, emb)) for emb in meta.get("embeddings", [])],
                encoders=[EncoderMetadata(**enc) for enc in meta.get("encoders", [])],
            )
            for name, meta in data.get("by_modality", {}).items()
        }
        fused = data.get("fused")
        return cls(
            message_id=data["message_id"],
            user_id=data["user_id"],
            fused=[[float(x) for x in emb] for emb in fused] if fused is not None else None,
            by_modality=by_modality,
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
            payload_keys = {"message_id", "user_id", "fused", "by_modality"}
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
class BDIIntentionPayload(EventPayload):
    """Payload representing a BDI intention."""

    goal: str
    priority: int


@dataclass
class WarningPayload(EventPayload):
    """Payload for ontology or system warnings."""

    message: str
    facts: Optional[list[tuple[str, str, str]]] = None
