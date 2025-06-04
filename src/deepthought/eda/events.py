"""
Event definitions for DeepThought reThought.

This module defines the event structures and naming conventions used
in the DeepThought reThought system's event-driven architecture.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import json


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
    def from_json(cls, json_str: str) -> 'EventPayload':
        """Create a payload instance from a JSON string."""
        data = json.loads(json_str)
        return cls(**data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EventPayload':
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
