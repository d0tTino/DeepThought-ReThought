from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np

dummy_cv2 = ModuleType("cv2")


class _DummyCapture:  # pragma: no cover - test shim
    def __init__(self, *args, **kwargs):
        self._released = False

    def get(self, *_args, **_kwargs):
        return 1.0

    def read(self):
        return False, None

    def release(self):  # pragma: no cover - shim
        self._released = True


dummy_cv2.CAP_PROP_FPS = 5
dummy_cv2.COLOR_BGR2RGB = 0
dummy_cv2.VideoCapture = lambda *args, **kwargs: _DummyCapture()
dummy_cv2.cvtColor = lambda frame, _code: frame

sys.modules.setdefault("cv2", dummy_cv2)

pil_module = ModuleType("PIL")
image_module = ModuleType("PIL.Image")
image_module.fromarray = lambda array: array
pil_module.Image = image_module
sys.modules.setdefault("PIL", pil_module)
sys.modules.setdefault("PIL.Image", image_module)

MODULE_NAME = "deepthought.services.perception.worker_video"
MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "deepthought"
    / "services"
    / "perception"
    / "worker_video.py"
)


def _load_worker_module() -> ModuleType:
    if MODULE_NAME in sys.modules:
        return sys.modules[MODULE_NAME]  # pragma: no cover - reuse cached module

    services_pkg = ModuleType("deepthought.services")
    services_pkg.__path__ = []  # type: ignore[attr-defined]
    perception_pkg = ModuleType("deepthought.services.perception")
    perception_pkg.__path__ = []  # type: ignore[attr-defined]
    services_pkg.perception = perception_pkg
    sys.modules["deepthought.services"] = services_pkg
    sys.modules["deepthought.services.perception"] = perception_pkg

    spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"Unable to load module spec for {MODULE_NAME}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    spec.loader.exec_module(module)
    perception_pkg.worker_video = module
    return module


_worker_module = _load_worker_module()
VideoPerceptionWorker = _worker_module.VideoPerceptionWorker


def test_video_perception_worker(monkeypatch):
    feats = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    times = np.array([0.0, 1.0], dtype=np.float32)

    def fake_grid(path, decode_fps, model_type, grid_fps, embed_cache=None):
        assert model_type == "siglip"
        return feats, times

    monkeypatch.setattr(_worker_module, "video_to_feature_grid", fake_grid)

    worker = VideoPerceptionWorker(decode_fps=1, model_type="siglip")
    out_feats, out_times = worker("dummy.mp4")

    assert np.array_equal(out_feats, feats)
    assert np.array_equal(out_times, times)


def test_video_perception_worker_with_revision(monkeypatch, tmp_path):
    feats = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    times = np.array([0.0, 1.0], dtype=np.float32)
    captured: dict[str, object] = {}

    def fake_grid(path, decode_fps, model_type, grid_fps, embed_cache=None):
        captured["args"] = {
            "path": path,
            "decode_fps": decode_fps,
            "model_type": model_type,
            "grid_fps": grid_fps,
            "embed_cache": embed_cache,
        }
        return feats, times

    monkeypatch.setattr(_worker_module, "video_to_feature_grid", fake_grid)

    cache_dir = tmp_path / "cache"
    worker = VideoPerceptionWorker(
        decode_fps=1,
        model_type="siglip@myrev",
        grid_fps=1,
        cache_dir=cache_dir,
    )

    video_path = tmp_path / "sample.mp4"
    out_feats, out_times = worker(video_path)

    assert np.array_equal(out_feats, feats)
    assert np.array_equal(out_times, times)

    args = captured["args"]
    assert isinstance(args, dict)
    assert args["model_type"] == "siglip"
    assert worker.model_type == "siglip@myrev"
    assert worker.model_revision == "myrev"

    feats_path = cache_dir / "sample_1_siglip@myrev_1_feats.npy"
    times_path = cache_dir / "sample_1_siglip@myrev_1_times.npy"
    embed_path = cache_dir / "sample_1_siglip@myrev_1_embed.npy"
    assert feats_path.exists()
    assert times_path.exists()
    assert args["embed_cache"] == embed_path
