"""
Event definitions for DeepThought reThought.

This module defines the event structures and naming conventions used
in the DeepThought reThought system's event-driven architecture.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
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

    # Social graph events
    SOCIAL_GRAPH_UPDATE = "dtr.social.graph.update"
    """Published when incremental updates to the social graph occur."""

    SOCIAL_GRAPH_SNAPSHOT = "dtr.social.graph.snapshot"
    """Published when a full snapshot of the social graph is available."""

    # Quest lifecycle events
    QUEST_CREATE = "dtr.quest.create"
    """Published when a new quest is created."""

    QUEST_UPDATE = "dtr.quest.update"
    """Published when quest metadata or progress is updated."""

    QUEST_DONE = "dtr.quest.done"
    """Published when a quest has been completed."""

    # Response selection events
    RESPONSE_CANDIDATES = "dtr.response.candidates"
    """Published with the set of response candidates produced by responders."""

    RESPONSE_RANKED = "dtr.response.ranked"
    """Published after ranking response candidates."""

    # Perception embedding events
    PERCEPTION_AUDIO_EMBED = "dtr.perception.audio.embed"
    """Published when audio perception embeddings are created."""

    PERCEPTION_IMAGE_EMBED = "dtr.perception.image.embed"
    """Published when image perception embeddings are created."""
    
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


@dataclass
class SocialGraphUpdatePayload(EventPayload):
    """Payload for incremental social graph updates."""

    user_id: str
    updates: Dict[str, Any]
    timestamp: Optional[str] = None


@dataclass
class SocialGraphSnapshotPayload(EventPayload):
    """Payload for full social graph snapshots."""

    user_id: str
    graph: Dict[str, Any]
    timestamp: Optional[str] = None


@dataclass
class QuestCreatePayload(EventPayload):
    """Payload for quest creation events."""

    quest_id: str
    name: str
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None


@dataclass
class QuestUpdatePayload(EventPayload):
    """Payload for quest update events."""

    quest_id: str
    status: str
    progress: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None


@dataclass
class QuestDonePayload(EventPayload):
    """Payload for quest completion events."""

    quest_id: str
    result: Dict[str, Any]
    timestamp: Optional[str] = None


@dataclass
class ResponseCandidatesPayload(EventPayload):
    """Payload for response candidate generation events."""

    input_id: str
    candidates: List[Dict[str, Any]]
    timestamp: Optional[str] = None


@dataclass
class ResponseRankedPayload(EventPayload):
    """Payload for response ranking events."""

    input_id: str
    ranked_candidates: List[Dict[str, Any]]
    selected_index: Optional[int] = None
    timestamp: Optional[str] = None


@dataclass
class PerceptionAudioEmbedPayload(EventPayload):
    """Payload for audio perception embedding events."""

    audio_id: str
    embedding: List[float]
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None


@dataclass
class PerceptionImageEmbedPayload(EventPayload):
    """Payload for image perception embedding events."""

    image_id: str
    embedding: List[float]
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None
