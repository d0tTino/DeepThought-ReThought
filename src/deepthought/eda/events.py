"""
Event definitions for DeepThought reThought.

This module defines the event structures and naming conventions used
in the DeepThought reThought system's event-driven architecture.
"""

import json
from dataclasses import dataclass
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
        return json.dumps(self.__dict__)

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
class BDIIntentionPayload(EventPayload):
    """Payload representing a BDI intention."""

    goal: str
    priority: int


@dataclass
class WarningPayload(EventPayload):
    """Payload for ontology or system warnings."""

    message: str
    facts: Optional[list[tuple[str, str, str]]] = None
