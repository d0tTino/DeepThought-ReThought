from __future__ import annotations

"""Simple risk scoring utilities."""

from .neuralsymbolic import security_hotfix


def score_commit_risk(message: str) -> float:
    """Return a risk score for ``message`` based on ``security_hotfix`` probability."""
    try:
        result = security_hotfix(message)
        if hasattr(result, "item"):
            return float(result.item())
        return float(result)
    except Exception:
        return 0.0
