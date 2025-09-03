"""DeepThought package initialization."""

from __future__ import annotations

import os
import sys
import types

try:  # Ensure prometheus_client is loaded before tests patch it
    import prometheus_client  # noqa: F401
except Exception:  # pragma: no cover - optional dependency may be missing
    pass

try:  # Ensure aiosqlite is loaded before tests patch it
    import aiosqlite  # noqa: F401
except Exception:  # pragma: no cover - optional dependency may be missing
    pass

__version__ = "0.1.0"

# Re-export modules subpackage for convenient access
from . import affinity  # noqa: F401

# Importing heavy submodules like ``goal_scheduler`` at module import time can
# trigger circular import errors during test collection.  Avoid eager imports and
# instead load them lazily when accessed.
if not os.environ.get("DEEPTHOUGHT_LIGHT_IMPORT"):
    import importlib

    def __getattr__(name: str):  # pragma: no cover - tiny wrapper
        if name in {"goal_scheduler", "harness", "learn"}:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
            return module
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# modules depends on optional external packages (e.g. nats). Import it lazily
if not os.environ.get("DEEPTHOUGHT_LIGHT_IMPORT"):
    try:  # pragma: no cover - optional dependency may be missing
        from . import modules  # type: ignore  # noqa: F401
        from . import train  # noqa: F401
    except Exception:  # pragma: no cover - optional dependency may be missing
        modules = None  # type: ignore
        train = None  # type: ignore
else:  # pragma: no cover - skip heavy optional import
    modules = None  # type: ignore
    train = None  # type: ignore
# motivate requires NATS, which may not be installed in test environments
try:  # pragma: no cover - optional dependency may be missing
    mod_name = __name__ + ".motivate"
    stub = sys.modules.get(mod_name)
    if isinstance(stub, types.ModuleType) and not getattr(stub, "__file__", None):
        sys.modules.pop(mod_name, None)
    from . import motivate  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover - optional dependency may be missing
    pass
try:  # pragma: no cover - optional dependency may be missing
    from . import orchestrator  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover - optional dependency may be missing
    pass
from . import persona  # noqa: F401
from . import utils  # noqa: F401

# neuromorphic uses optional nengo dependency
try:  # pragma: no cover - optional dependency may be missing
    from . import neuromorphic  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover - optional dependency may be missing
    neuromorphic = None  # type: ignore

# risk scoring utilities rely only on builtin dependencies
from . import risk  # noqa: F401
