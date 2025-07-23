from __future__ import annotations

"""Utilities for neural-symbolic predicates."""

from typing import Any, Callable

try:
    from deepproblog.logic import WrappedFunction  # type: ignore
    _USE_DEEPPROBLOG = True
except Exception:  # pragma: no cover - optional dependency
    _USE_DEEPPROBLOG = False

try:
    import ltn  # type: ignore
    _USE_LTN = True
except Exception:  # pragma: no cover - optional dependency
    _USE_LTN = False


class NeuralPredicate:
    """Wrapper around a Python callable exposed as a neural predicate."""

    def __init__(self, name: str, model_func: Callable[[Any], float]) -> None:
        self.name = name
        self.model_func = model_func
        self.predicate = self._build()

    def _build(self) -> Callable[..., Any]:
        if _USE_DEEPPROBLOG:
            def wrapper(term: Any) -> float:
                arg = str(term.args[0]) if hasattr(term, "args") else str(term)
                return float(self.model_func(arg))
            return WrappedFunction(self.name, 1, wrapper)
        if _USE_LTN:
            return ltn.Predicate(self.model_func)
        return self.model_func

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.predicate(*args, **kwargs)


def _security_hotfix_model(message: str) -> float:
    """Heuristic model returning ``1.0`` if ``message`` looks like a security hotfix."""
    keywords = ["security", "cve", "vulnerability", "hotfix", "patch"]
    lowered = message.lower()
    return 1.0 if any(k in lowered for k in keywords) else 0.0


security_hotfix = NeuralPredicate("security_hotfix", _security_hotfix_model)
