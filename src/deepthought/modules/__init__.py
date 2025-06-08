"""
DeepThought reThought system modules.

This package contains the main functional modules of the DeepThought reThought system,
including input handling, memory, LLM processing, and output handling components.
"""

from .input_handler import InputHandler
from .output_handler import OutputHandler
from .memory_stub import MemoryStub
from .memory_basic import BasicMemory
from .llm_stub import LLMStub

# BasicLLM has heavy optional dependencies (transformers/torch). Import it lazily
# so modules that do not require those packages can still be used.
try:  # pragma: no cover - optional dependency
    from .llm_basic import BasicLLM  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency may be missing

    _missing_deps_err = exc

    class BasicLLM:  # type: ignore
        """Placeholder that raises if instantiated when deps are missing."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ImportError(
                "BasicLLM requires optional dependencies (transformers, torch)"
            ) from _missing_deps_err


__all__ = [
    "InputHandler",
    "OutputHandler",
    "MemoryStub",
    "LLMStub",
    "BasicMemory",
    "BasicLLM",
]
