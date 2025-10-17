"""Helpers for perception utilities.

This package is occasionally stubbed by tests to avoid importing heavy
dependencies. Some of those stubs expose only the pieces required for the
specific test case which can break other suites that expect the full
``worker_video`` implementation. To keep import order from mattering we detect
such partial stubs and graft the real implementation's symbols onto them.
"""

from __future__ import annotations

import sys
from importlib import util as importlib_util
from pathlib import Path
from types import ModuleType
from typing import Iterable


def _load_worker_video_impl() -> ModuleType | None:
    """Return the real ``worker_video`` module loaded under an auxiliary name."""

    module_name = f"{__name__}._worker_video_impl"
    module = sys.modules.get(module_name)
    if module is not None:
        return module
    path = Path(__file__).with_name("worker_video.py")
    spec = importlib_util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - unexpected
        return None
    module = importlib_util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:  # pragma: no cover - import errors bubble up later
        return None
    sys.modules[module_name] = module
    return module


def _ensure_worker_video_exports() -> None:
    """Populate stubs with the real worker video helpers when required."""

    name = f"{__name__}.worker_video"
    module = sys.modules.get(name)
    if module is None:
        return
    if getattr(module, "__file__", None):
        return  # Real module already imported

    impl = _load_worker_video_impl()
    if impl is None:
        return

    required: Iterable[str] = (
        "decode_video",
        "embed_frames",
        "interpolate_features",
        "video_to_feature_grid",
    )
    for attr in required:
        value = getattr(impl, attr, None)
        if value is not None:
            setattr(module, attr, value)


_ensure_worker_video_exports()
