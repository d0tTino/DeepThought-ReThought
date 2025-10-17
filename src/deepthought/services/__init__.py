"""Service utilities for DeepThought.

This module previously imported every service eagerly which pulled in heavy
dependencies such as :mod:`aiosqlite`, transformer models, or optional
perception workers.  Many of those dependencies are not available in the
lightweight unit test environment which meant ``import deepthought.services``
could fail before tests had a chance to skip.  To keep imports cheap and
robust we expose the public services through a small lazy importer.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "FileGraphDAL",
    "CognitiveCoreService",
    "PersonaManager",
    "DBManager",
    "TrustService",
    "EngagementPolicy",
    "should_reply",
    "SocialGraphService",
    "PlanningService",
    "ReasoningService",
    "manipulation_score",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "FileGraphDAL": ("file_graph_dal", "FileGraphDAL"),
    "CognitiveCoreService": ("cognitive_core_service", "CognitiveCoreService"),
    "PersonaManager": ("persona_manager", "PersonaManager"),
    "DBManager": ("db_manager", "DBManager"),
    "TrustService": ("trust_service", "TrustService"),
    "EngagementPolicy": ("engagement_policy", "EngagementPolicy"),
    "should_reply": ("engagement_policy", "should_reply"),
    "SocialGraphService": ("social_graph_service", "SocialGraphService"),
    "PlanningService": ("planning_service", "PlanningService"),
    "ReasoningService": ("reasoning_service", "ReasoningService"),
    "manipulation_score": ("manipulative_detection", "manipulation_score"),
}


def _load_attribute(module_name: str, attr_name: str, export_name: str) -> Any:
    """Import ``module_name`` from this package and return ``attr_name``."""

    module = import_module(f"{__name__}.{module_name}")
    attr = getattr(module, attr_name)
    globals()[attr_name] = attr
    if export_name != attr_name:
        globals()[export_name] = attr
    return attr


def __getattr__(name: str) -> Any:  # pragma: no cover - exercised indirectly
    try:
        module_name, attr_name = _LAZY_IMPORTS[name]
    except KeyError as exc:  # pragma: no cover - defensive programming
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'") from exc
    return _load_attribute(module_name, attr_name, name)


def __dir__() -> list[str]:  # pragma: no cover - simple helper
    return sorted(set(__all__ + list(globals().keys())))
