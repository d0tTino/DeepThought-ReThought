"""DeepThought package initialization."""

from __future__ import annotations

import importlib
import os

__version__ = "0.1.0"

# Re-export modules subpackage for convenient access
from . import affinity  # noqa: F401
from . import goal_scheduler  # noqa: F401
from . import harness  # noqa: F401
from . import learn  # noqa: F401
# modules depends on optional external packages (e.g. nats). Import it lazily
if not os.environ.get("DEEPTHOUGHT_LIGHT_IMPORT"):
    try:  # pragma: no cover - optional dependency may be missing
        from . import modules  # type: ignore  # noqa: F401
        from . import train_script  # noqa: F401
    except Exception:  # pragma: no cover - optional dependency may be missing
        modules = None  # type: ignore
        train_script = None  # type: ignore
else:  # pragma: no cover - skip heavy optional import
    modules = None  # type: ignore
    train_script = None  # type: ignore
# motivate requires NATS, which may not be installed in test environments
try:  # pragma: no cover - optional dependency may be missing
    from . import motivate  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover - optional dependency may be missing
    motivate = None  # type: ignore
from . import persona  # noqa: F401
# neuromorphic uses optional nengo dependency
try:  # pragma: no cover - optional dependency may be missing
    from . import neuromorphic  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover - optional dependency may be missing
    neuromorphic = None  # type: ignore

