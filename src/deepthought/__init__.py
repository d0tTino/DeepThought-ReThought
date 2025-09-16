"""DeepThought package initialization."""

from __future__ import annotations

import os
import sys
import types


def _ensure_torch_numpy_compat() -> None:
    """Provide a fallback ``torch.from_numpy`` when NumPy bindings are missing."""

    try:  # pragma: no cover - optional dependency may be missing
        import torch
        import numpy as _np
    except Exception:
        return

    if getattr(torch, "_deepthought_numpy_checked", False):  # pragma: no branch - simple guard
        return

    torch._deepthought_numpy_checked = True  # type: ignore[attr-defined]

    try:
        torch.from_numpy(_np.zeros(1, dtype=_np.float32))
    except RuntimeError as exc:
        if "Numpy is not available" not in str(exc):
            return

        dtype_map = {
            _np.dtype(_np.float32): torch.float32,
            _np.dtype(_np.float64): torch.float64,
            _np.dtype(_np.float16): torch.float16,
            _np.dtype(_np.int64): torch.int64,
            _np.dtype(_np.int32): torch.int32,
            _np.dtype(_np.int16): torch.int16,
            _np.dtype(_np.int8): torch.int8,
            _np.dtype(_np.uint8): torch.uint8,
            _np.dtype(_np.uint16): torch.int32,
            _np.dtype(_np.uint32): torch.int64,
            _np.dtype(_np.uint64): torch.int64,
            _np.dtype(_np.bool_): torch.bool,
            _np.dtype(_np.complex64): torch.complex64,
            _np.dtype(_np.complex128): torch.complex128,
        }
        if hasattr(_np, "float128"):
            dtype_map[_np.dtype(_np.float128)] = torch.float64

        def _from_numpy_fallback(array: object):  # pragma: no cover - exercised indirectly in tests
            if isinstance(array, (list, tuple)) and len(array) == 1 and isinstance(
                array[0], (_np.ndarray, _np.memmap)
            ):
                array = array[0]
            arr = _np.asarray(array)
            dtype = dtype_map.get(arr.dtype)
            if arr.ndim == 0:
                value = arr.item()
                if dtype is None:
                    return torch.tensor(value)
                return torch.tensor(value, dtype=dtype)
            data = _np.array(arr, copy=True)
            if dtype is None:
                tensor = torch.tensor(data)
            else:
                tensor = torch.tensor(data, dtype=dtype)
            if tensor.shape != arr.shape:
                tensor = tensor.reshape(arr.shape)
            return tensor

        torch.from_numpy = _from_numpy_fallback  # type: ignore[assignment]

        tensor_dtype_map = {
            torch.float32: _np.float32,
            torch.float64: _np.float64,
            torch.float16: _np.float16,
            torch.int64: _np.int64,
            torch.int32: _np.int32,
            torch.int16: _np.int16,
            torch.int8: _np.int8,
            torch.uint8: _np.uint8,
            torch.bool: _np.bool_,
            torch.complex64: _np.complex64,
            torch.complex128: _np.complex128,
        }

        def _tensor_numpy(self):  # pragma: no cover - exercised indirectly in tests
            array = self.detach().cpu()
            dtype = tensor_dtype_map.get(array.dtype)
            data = array.tolist()
            if dtype is None:
                return _np.asarray(data)
            return _np.asarray(data, dtype=dtype)

        torch.Tensor.numpy = _tensor_numpy  # type: ignore[assignment]


_ensure_torch_numpy_compat()

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
from . import affinity  # noqa: F401,E402

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
from . import persona  # noqa: F401,E402
from . import utils  # noqa: F401,E402

# neuromorphic uses optional nengo dependency
try:  # pragma: no cover - optional dependency may be missing
    from . import neuromorphic  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover - optional dependency may be missing
    neuromorphic = None  # type: ignore

# risk scoring utilities rely only on builtin dependencies
from . import risk  # noqa: F401,E402
