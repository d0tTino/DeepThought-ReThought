"""DeepThought package initialization."""

from __future__ import annotations

import os
import sys
import types

try:  # Ensure prometheus_client is loaded before tests patch it
    import prometheus_client  # noqa: F401
except Exception:  # pragma: no cover - optional dependency may be missing
    pass

__version__ = "0.1.0"

# Re-export modules subpackage for convenient access
from . import affinity  # noqa: F401
if not os.environ.get("DEEPTHOUGHT_LIGHT_IMPORT"):
    from . import goal_scheduler  # noqa: F401
    from . import harness  # noqa: F401
    from . import learn  # noqa: F401
else:  # pragma: no cover - skip heavy optional imports
    goal_scheduler = None  # type: ignore
    harness = None  # type: ignore
    learn = None  # type: ignore


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
