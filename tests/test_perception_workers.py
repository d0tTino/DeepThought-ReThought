import os
import sys
import types
from pathlib import Path

os.environ.setdefault("DEEPTHOUGHT_LIGHT_IMPORT", "1")

import importlib.util

import numpy as np
import pytest
from scipy.io import wavfile

_stub_aiosqlite = types.ModuleType("aiosqlite")
_stub_aiosqlite.connect = None  # type: ignore[attr-defined]
sys.modules.setdefault("aiosqlite", _stub_aiosqlite)

_stub_pyperplan = types.ModuleType("pyperplan")
_stub_pyperplan_pddl = types.ModuleType("pyperplan.pddl")
_stub_pyperplan_parser = types.ModuleType("pyperplan.pddl.parser")
_stub_pyperplan_planner = types.ModuleType("pyperplan.planner")
_stub_pyperplan_search = types.ModuleType("pyperplan.search")

_stub_pyperplan.__path__ = []  # type: ignore[attr-defined]


class _DummyParser:  # pragma: no cover - never executed in these tests
    def __init__(self, *args: object, **kwargs: object) -> None:
        raise RuntimeError("pyperplan is not available in the test environment")


_stub_pyperplan_parser.Parser = _DummyParser  # type: ignore[attr-defined]
_stub_pyperplan_pddl.parser = _stub_pyperplan_parser  # type: ignore[attr-defined]
_stub_pyperplan.pddl = _stub_pyperplan_pddl  # type: ignore[attr-defined]
_stub_pyperplan_planner._ground = lambda problem: problem  # type: ignore[attr-defined]
_stub_pyperplan_search.breadth_first_search = lambda task: []  # type: ignore[attr-defined]

sys.modules.setdefault("pyperplan", _stub_pyperplan)
sys.modules.setdefault("pyperplan.pddl", _stub_pyperplan_pddl)
sys.modules.setdefault("pyperplan.pddl.parser", _stub_pyperplan_parser)
sys.modules.setdefault("pyperplan.planner", _stub_pyperplan_planner)
sys.modules.setdefault("pyperplan.search", _stub_pyperplan_search)
sys.modules.setdefault("cv2", types.ModuleType("cv2"))

_stub_pil = types.ModuleType("PIL")
_stub_pil_image = types.ModuleType("PIL.Image")
_stub_pil.Image = _stub_pil_image  # type: ignore[attr-defined]
sys.modules.setdefault("PIL", _stub_pil)
sys.modules.setdefault("PIL.Image", _stub_pil_image)

_services_stub = types.ModuleType("deepthought.services")
_services_stub.__path__ = [
    str(Path(__file__).resolve().parents[1] / "src" / "deepthought" / "services")
]
sys.modules.setdefault("deepthought.services", _services_stub)

import deepthought  # noqa: E402  # ensure the root package is loaded

setattr(deepthought, "services", _services_stub)

_perception_stub = types.ModuleType("deepthought.services.perception")
_perception_stub.__path__ = [
    str(Path(__file__).resolve().parents[1] / "src" / "deepthought" / "services" / "perception")
]
sys.modules.setdefault("deepthought.services.perception", _perception_stub)
setattr(_services_stub, "perception", _perception_stub)


def _load_perception_module(module: str):
    module_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "deepthought"
        / "services"
        / "perception"
        / f"{module}.py"
    )
    full_name = f"deepthought.services.perception.{module}"
    spec = importlib.util.spec_from_file_location(full_name, module_path)
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = loaded
    spec.loader.exec_module(loaded)
    setattr(_perception_stub, module, loaded)
    return loaded


config_module = _load_perception_module("config")
worker_audio_module = _load_perception_module("worker_audio")
worker_text_module = _load_perception_module("worker_text")
worker_video_module = _load_perception_module("worker_video")
asr_module = _load_perception_module("asr")

PerceptionConfig = config_module.PerceptionConfig
AudioPerceptionWorker = worker_audio_module.AudioPerceptionWorker
TextPerceptionWorker = worker_text_module.TextPerceptionWorker
VideoPerceptionWorker = worker_video_module.VideoPerceptionWorker
transcribe_audio_tokens = asr_module.transcribe_audio_tokens


class _DummySentenceModel:
    def __init__(self) -> None:
        self.last_text = None

    def encode(self, text: str) -> np.ndarray:
        self.last_text = text
        length = len(text)
        return np.asarray([length, length + 1], dtype=np.float32)


def test_audio_perception_worker(tmp_path):
    sr = 16000
    data = np.ones(int(0.05 * sr), dtype=np.int16)
    path = tmp_path / "test.wav"
    wavfile.write(path, sr, data)

    worker = AudioPerceptionWorker(window_size=0.02, step_size=0.01)
    features, times = worker(path)

    assert features.shape == (4, 4)
    assert times.shape == (4, 2)


def test_text_perception_worker(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "deepthought.services.perception.worker_text.SentenceTransformer",
        lambda name: _DummySentenceModel(),
    )
    tokens = [("hi", 0.0, 0.05), ("there", 0.05, 0.1)]
    memmap_path = tmp_path / "tokens.dat"

    worker = TextPerceptionWorker(model_name="dummy", hop_seconds=0.05)
    feats, times = worker(tokens, memmap_path)

    assert feats.shape == (2, 2)
    assert times.shape == (2, 2)
    assert np.allclose(feats[0], [2, 3])
    assert np.allclose(feats[1], [5, 6])


def test_video_perception_worker(monkeypatch):
    dummy_feats = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    dummy_times = np.array([0.0, 1.0], dtype=np.float32)

    monkeypatch.setattr(
        "deepthought.services.perception.worker_video.video_to_feature_grid",
        lambda path, decode_fps, model_type, grid_fps: (dummy_feats, dummy_times),
    )

    worker = VideoPerceptionWorker(decode_fps=1, model_type="siglip")
    feats, times = worker("dummy.mp4")

    assert np.array_equal(feats, dummy_feats)
    assert np.array_equal(times, dummy_times)


def test_text_perception_worker_redacts_pii(monkeypatch, tmp_path):
    model = _DummySentenceModel()
    monkeypatch.setattr(
        "deepthought.services.perception.worker_text.SentenceTransformer",
        lambda name: model,
    )
    token = [("Email me at test@example.com or call 123-456-7890", 0.0, 0.05)]
    memmap_path = tmp_path / "pii_tokens.dat"

    worker = TextPerceptionWorker(model_name="dummy", hop_seconds=0.05)
    feats, _ = worker(token, memmap_path)

    expected = "Email me at [REDACTED] or call [REDACTED]"
    assert model.last_text == expected
    exp_len = len(expected)
    assert np.allclose(feats[0], [exp_len, exp_len + 1])


def test_asr_transcribe_audio_tokens_caches(monkeypatch, tmp_path):
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"RIFF")

    load_calls: list[tuple[str, str | None]] = []

    class _DummyModel:
        def __init__(self) -> None:
            self.calls = 0

        def transcribe(self, path: str, **_: object):
            self.calls += 1
            return {
                "language": "en",
                "segments": [
                    {
                        "start": 0.0,
                        "end": 1.0,
                        "text": "hello world",
                        "words": [
                            {"word": "hello", "start": 0.0, "end": 0.5},
                            {"word": "world", "start": 0.5, "end": 1.0},
                        ],
                    }
                ],
            }

    dummy_model = _DummyModel()

    def _load_model(name: str, *, device: str | None = None, **_: object):
        load_calls.append((name, device))
        return dummy_model

    whisper_module = types.ModuleType("whisper")
    whisper_module.load_model = _load_model  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "whisper", whisper_module)

    from deepthought.services.perception import asr as asr_module

    asr_module._MODEL_CACHE.clear()

    tokens = transcribe_audio_tokens(audio_path, cache_dir=tmp_path, model_name="small", language="en")
    assert tokens == [("hello", 0.0, 0.5), ("world", 0.5, 1.0)]
    assert dummy_model.calls == 1
    assert load_calls == [("small", None)]

    # Second call should reuse the cached transcript and avoid another model invocation.
    cached_tokens = transcribe_audio_tokens(audio_path, cache_dir=tmp_path, model_name="small", language="en")
    assert cached_tokens == tokens
    assert dummy_model.calls == 1
    assert load_calls == [("small", None)]

    cfg = PerceptionConfig()
    expected_cache = (
        tmp_path
        / f"{audio_path.stem}_{cfg.audio_model}_ws{cfg.audio_window_size}_ss{cfg.audio_hop_size}.transcript.json"
    )
    assert expected_cache.exists()

    asr_module._MODEL_CACHE.clear()
