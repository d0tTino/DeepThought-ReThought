from __future__ import annotations

"""Skeleton implementation of online LoRA learning."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Tuple


@dataclass
class Interaction:
    """A single prompt/response/reward tuple."""

    prompt: str
    response: str
    reward: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


class OnlineLoRALearner:
    """Minimal learner storing interactions per guild."""

    def __init__(self, model_name: str = "tiny") -> None:
        self.model_name = model_name
        self.memory: Dict[str, List[Interaction]] = {}

    def record_interaction(self, guild_id: str, prompt: str, response: str, reward: float) -> None:
        """Store a training example for a guild."""
        self.memory.setdefault(guild_id, []).append(Interaction(prompt=prompt, response=response, reward=reward))

    def get_training_data(self, guild_id: str) -> List[Interaction]:
        """Return stored interactions for a guild."""
        return list(self.memory.get(guild_id, []))

    def clear_data(self, guild_id: str) -> None:
        """Clear stored interactions after an update."""
        self.memory.pop(guild_id, None)
