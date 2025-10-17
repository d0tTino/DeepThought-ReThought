"""Perception utilities for DeepThought services.

The perception stack pulls in optional dependencies such as Whisper, CLIP,
and Sentence Transformers.  Importing everything eagerly makes the package
fragile in unit tests where those extras are intentionally absent.  Similar to
``deepthought.services`` we provide a thin lazy importer so that individual
workers are only loaded when accessed.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "TextPerceptionWorker",
    "AudioPerceptionWorker",
    "transcribe_audio_tokens",
    "PerceptionService",
    "PerceptionPublisher",
    "UserEmbeddings",
    "PerceptionConfig",
    "VideoPerceptionWorker",
    "main",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "transcribe_audio_tokens": ("asr", "transcribe_audio_tokens"),
    "main": ("cli", "main"),
    "PerceptionConfig": ("config", "PerceptionConfig"),
    "PerceptionPublisher": ("publisher", "PerceptionPublisher"),
    "PerceptionService": ("service", "PerceptionService"),
    "UserEmbeddings": ("user_embeddings", "UserEmbeddings"),
    "AudioPerceptionWorker": ("worker_audio", "AudioPerceptionWorker"),
    "TextPerceptionWorker": ("worker_text", "TextPerceptionWorker"),
    "VideoPerceptionWorker": ("worker_video", "VideoPerceptionWorker"),
}


def _load_attribute(module_name: str, attr_name: str, export_name: str) -> Any:
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
