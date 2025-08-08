"""Quest management utilities."""

from .storage import (
    Quest,
    Objective,
    Evidence,
    Epiphany,
    LieRecord,
    QuestStorage,
)

from . import dsl
from .templates import (
    QuestTemplate,
    CooldownTracker,
    bind_slot,
    auto_spawn_quests,
    TEMPLATES,
)

__all__ = [
    "Quest",
    "Objective",
    "Evidence",
    "Epiphany",
    "LieRecord",
    "QuestStorage",
    "dsl",
    "QuestTemplate",
    "CooldownTracker",
    "bind_slot",
    "auto_spawn_quests",
    "TEMPLATES",
]
