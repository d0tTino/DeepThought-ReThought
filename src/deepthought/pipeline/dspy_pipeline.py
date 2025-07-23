from __future__ import annotations

"""Utilities for building DSPy question answering pipelines."""

from types import SimpleNamespace
from typing import Callable

try:  # pragma: no cover - optional dependency
    import dspy
except Exception:  # pragma: no cover - dspy may not be installed
    dspy = None  # type: ignore


def build_qa_pipeline() -> Callable[[str], str]:
    """Return a function that answers questions using DSPy."""

    if dspy is None:  # pragma: no cover - tested via mock
        raise ImportError("DSPy library is required")

    class QASignature(dspy.Signature):  # type: ignore[attr-defined]
        """Signature for simple question answering."""

        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    qa_func = dspy.LMFunction(QASignature)

    def pipeline(question: str) -> str:
        result = qa_func(question=question)
        answer = getattr(result, "answer", None)
        if not isinstance(answer, str):
            raise ValueError("DSPy pipeline returned invalid result")
        return answer

    return pipeline
