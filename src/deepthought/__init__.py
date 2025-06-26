"""DeepThought package initialization."""

__version__ = "0.1.0"

# Re-export modules subpackage for convenient access
from . import harness  # noqa: F401
from . import learn  # noqa: F401
from . import modules  # noqa: F401

try:
    from . import motivate  # noqa: F401
except Exception:  # pragma: no cover - optional dependency may be missing
    pass
