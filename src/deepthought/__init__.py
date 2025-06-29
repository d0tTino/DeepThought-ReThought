"""DeepThought package initialization."""

from __future__ import annotations

import importlib

__version__ = "0.1.0"

# Lazily expose common subpackages without importing heavy dependencies on
# module import.  ``__getattr__`` performs the actual import when an attribute is
# accessed.  This keeps startup lightweight for tests that only need a subset of
# the package.

__all__ = [
    "affinity",
    "goal_scheduler",
    "harness",
    "learn",
    "modules",
    "motivate",
    "persona",
]


def __getattr__(name: str) -> object:
    if name in __all__:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
