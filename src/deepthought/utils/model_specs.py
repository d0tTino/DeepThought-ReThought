"""Utilities for working with model specification strings."""

from __future__ import annotations

from typing import Tuple


def split_model_revision(spec: str) -> Tuple[str, str | None]:
    """Return ``(model_key, revision)`` parsed from ``spec``.

    ``spec`` strings may optionally include a ``@revision`` suffix. When no
    suffix is present the revision component is ``None``. Whitespace is
    preserved to avoid accidentally changing model identifiers.
    """

    if "@" not in spec:
        return spec, None
    model_key, revision = spec.split("@", 1)
    return model_key, revision or None
