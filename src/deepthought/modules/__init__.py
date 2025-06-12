"""
DeepThought reThought system modules.

This package contains the main functional modules of the DeepThought reThought system,
including input handling, memory, LLM processing, and output handling components.
"""

from .input_handler import InputHandler
from .memory_basic import BasicMemory
from .memory_graph import GraphMemory
from .memory_stub import MemoryStub
from .output_handler import OutputHandler

# KnowledgeGraphMemory requires mgclient which may be optional
try:  # pragma: no cover - optional dependency may be missing
    from .memory_kg import KnowledgeGraphMemory  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency may be missing
    _missing_kg_err = exc

    class KnowledgeGraphMemory:  # type: ignore
        """Placeholder that raises if instantiated when mgclient is missing."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ImportError("KnowledgeGraphMemory requires optional dependency mgclient") from _missing_kg_err


from .llm_stub import LLMStub

# ProductionLLM depends on additional optional packages (transformers, torch, peft)
try:  # pragma: no cover - optional dependency
    from .llm_prod import ProductionLLM  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency may be missing
    _missing_prod_err = exc

    class ProductionLLM:  # type: ignore
        """Placeholder that raises if instantiated when deps are missing."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ImportError(
                "ProductionLLM requires optional dependencies (transformers, torch, peft)"
            ) from _missing_prod_err


# BasicLLM has heavy optional dependencies (transformers/torch). Import it lazily
# so modules that do not require those packages can still be used.
try:  # pragma: no cover - optional dependency
    from .llm_basic import BasicLLM  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency may be missing

    _missing_deps_err = exc

    class BasicLLM:  # type: ignore
        """Placeholder that raises if instantiated when deps are missing."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ImportError("BasicLLM requires optional dependencies (transformers, torch)") from _missing_deps_err


__all__ = [
    "InputHandler",
    "OutputHandler",
    "MemoryStub",
    "LLMStub",
    "BasicMemory",
    "GraphMemory",
    "KnowledgeGraphMemory",
    "BasicLLM",
    "ProductionLLM",
]
