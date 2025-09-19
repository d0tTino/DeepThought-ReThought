"""Integration test covering audio-only perception with ASR tokens."""

from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.io import wavfile


# Many perception modules pull in heavy optional dependencies (NATS, OpenCV,
# sentence-transformers, etc.). Provide lightweight stubs so we can import the
# perception service without requiring those packages during test collection.
os.environ.setdefault("DEEPTHOUGHT_LIGHT_IMPORT", "1")


def _ensure_stub(module_name: str, module: types.ModuleType) -> None:
    """Register ``module`` under ``module_name`` if not already present."""

    sys.modules.setdefault(module_name, module)


# Stub sentence_transformers so TextPerceptionWorker can be imported without the
# real library. The test later monkeypatches this placeholder with a dummy
# encoder implementation.
sent_mod = types.ModuleType("sentence_transformers")


class _StubSentenceTransformer:  # pragma: no cover - only used if monkeypatching fails
    def __init__(self, *args, **kwargs) -> None:  # noqa: D401 - minimal stub
        raise RuntimeError("SentenceTransformer stub requires monkeypatch in tests")

    def encode(self, text: str) -> np.ndarray:
        raise RuntimeError("SentenceTransformer stub requires monkeypatch in tests")


sent_mod.SentenceTransformer = _StubSentenceTransformer
_ensure_stub("sentence_transformers", sent_mod)


# Provide stub packages for deepthought.modules and deepthought.services so
# importing submodules does not execute their heavyweight __init__ logic.
modules_pkg = types.ModuleType("deepthought.modules")
modules_pkg.__path__ = ["src/deepthought/modules"]
_ensure_stub("deepthought.modules", modules_pkg)

services_pkg = types.ModuleType("deepthought.services")
services_pkg.__path__ = ["src/deepthought/services"]
_ensure_stub("deepthought.services", services_pkg)

perception_pkg = types.ModuleType("deepthought.services.perception")
perception_pkg.__path__ = ["src/deepthought/services/perception"]
_ensure_stub("deepthought.services.perception", perception_pkg)
services_pkg.perception = perception_pkg  # type: ignore[attr-defined]


import deepthought  # noqa: E402

deepthought.services = services_pkg  # type: ignore[attr-defined]
deepthought.modules = modules_pkg  # type: ignore[attr-defined]


# Avoid importing OpenCV by stubbing the video worker module. The perception
# service only inspects the class reference, so a minimal placeholder suffices.
video_stub = types.ModuleType("deepthought.services.perception.worker_video")


class _StubVideoWorker:  # pragma: no cover - instantiation would be an error
    def __call__(self, path: str) -> None:
        raise RuntimeError("VideoPerceptionWorker should not be invoked in tests")


video_stub.VideoPerceptionWorker = _StubVideoWorker
_ensure_stub("deepthought.services.perception.worker_video", video_stub)


# Expose ModalityFuser from its concrete implementation on the stub package so
# perception.service can import it normally.
fuser_module = importlib.import_module("deepthought.modules.fuser")
modules_pkg.ModalityFuser = fuser_module.ModalityFuser
sys.modules.setdefault("deepthought.modules.fuser", fuser_module)


from deepthought.modules import ModalityFuser  # noqa: E402  (after stubs)
from deepthought.services.perception.service import PerceptionService  # noqa: E402
from deepthought.services.perception.worker_audio import AudioPerceptionWorker  # noqa: E402
from deepthought.services.perception.worker_text import TextPerceptionWorker  # noqa: E402


class DummyPublisher:
    """Capture publish payloads emitted by the perception service."""

    def __init__(self) -> None:
        self.kwargs: dict | None = None

    async def publish(self, **kwargs) -> None:  # pragma: no cover - trivial
        self.kwargs = kwargs


def _simple_asr_tokens(audio_path: Path, transcript: str) -> tuple[list[tuple[str, float, float]], float]:
    """Split ``transcript`` uniformly across the duration of ``audio_path``."""

    sr, samples = wavfile.read(audio_path)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    duration = samples.shape[0] / float(sr)
    words = transcript.split()
    if not words:
        raise ValueError("transcript must contain at least one word")
    hop = duration / len(words)
    tokens: list[tuple[str, float, float]] = []
    start = 0.0
    for idx, word in enumerate(words):
        end = start + hop
        if idx == len(words) - 1:
            end = duration
        tokens.append((word, float(start), float(end)))
        start = end
    return tokens, float(duration)


@pytest.mark.asyncio
async def test_perception_asr_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end audio perception publishes fused embeddings and spans."""

    if not hasattr(torch, "SymBool"):
        pytest.skip("PyTorch build lacks SymBool which ModalityFuser requires")

    class DummySentenceModel:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def encode(self, text: str) -> np.ndarray:
            self.calls.append(text)
            length = float(len(text))
            checksum = float(sum(ord(ch) for ch in text) % 997)
            return np.array([length, checksum], dtype=np.float32)

    model = DummySentenceModel()
    monkeypatch.setattr(
        "deepthought.services.perception.worker_text.SentenceTransformer",
        lambda name: model,
    )

    def dummy_select(_model: str, _model_path: str | Path | None, _sr: int):
        def embed(window: np.ndarray) -> np.ndarray:
            if window.size == 0:
                return np.zeros(4, dtype=np.float32)
            return np.array(
                [
                    float(window.mean()),
                    float(window.std()),
                    float(window.min()),
                    float(window.max()),
                ],
                dtype=np.float32,
            )

        return embed

    monkeypatch.setattr(
        "deepthought.perception.worker_audio._select_embedding_fn",
        dummy_select,
    )

    settings_stub = SimpleNamespace(
        wandb_enabled=False,
        wandb_project=None,
        wandb_sweep_id=None,
        wandb_upload_artifacts=False,
    )
    monkeypatch.setattr(
        "deepthought.services.perception.service.get_settings",
        lambda: settings_stub,
    )
    monkeypatch.setattr(
        "deepthought.perception.worker_audio.get_settings",
        lambda: settings_stub,
    )

    sr = 16_000
    duration = 0.3
    samples_per_word = int(sr * duration / 2)
    segment1 = 0.3 * np.sin(2 * np.pi * 220 * np.arange(samples_per_word) / sr)
    segment2 = 0.3 * np.sin(2 * np.pi * 440 * np.arange(samples_per_word) / sr)
    waveform = np.concatenate([segment1, segment2]).astype(np.float32)
    audio_path = tmp_path / "clip.wav"
    wavfile.write(audio_path, sr, waveform)

    transcript = "hello world"
    tokens, audio_duration = _simple_asr_tokens(audio_path, transcript)
    assert tokens and len(tokens) == len(transcript.split())
    assert tokens[0][1] == pytest.approx(0.0)
    assert tokens[-1][2] == pytest.approx(audio_duration)

    text_worker = TextPerceptionWorker(model_name="dummy", hop_seconds=0.05)
    audio_worker = AudioPerceptionWorker(
        window_size=0.05,
        step_size=0.05,
        model="dummy",
        cache_dir=tmp_path,
    )

    fuser = ModalityFuser({"text": 2, "audio": 4}, fused_dim=5)
    fuser.eval()

    fake_cfg = SimpleNamespace(
        grid_hop_size=text_worker.hop_seconds,
        text_cache_dir=None,
        audio_cache_dir=str(tmp_path),
        video_cache_dir=None,
        audio_window_size=audio_worker.window_size,
        audio_hop_size=audio_worker.step_size,
        audio_model=audio_worker.model,
        audio_model_path=None,
        text_model="dummy",
        text_hop_size=text_worker.hop_seconds,
        video_model="dummy",
        video_hop_size=1.0,
        modality_dims={"text": 2, "audio": 4},
    )
    monkeypatch.setattr(
        "deepthought.services.perception.service.PerceptionConfig",
        lambda: fake_cfg,
    )

    publisher = DummyPublisher()
    service = PerceptionService(
        publisher,
        text_worker=text_worker,
        audio_worker=audio_worker,
        fuser=fuser,
    )

    await service.run(
        message_id="msg-1",
        user_id="user-42",
        text_tokens=tokens,
        audio_path=str(audio_path),
        audio_opt_in=True,
    )

    assert publisher.kwargs is not None
    fused = publisher.kwargs["fused"]
    assert fused is not None
    fused_array = np.asarray(fused, dtype=np.float32)

    span_count = int(np.ceil(audio_duration / text_worker.hop_seconds))
    assert fused_array.shape == (span_count, fuser.project.out_features)
    assert not np.allclose(fused_array, 0.0)

    modalities = publisher.kwargs["by_modality"]
    assert set(modalities.keys()) >= {"text", "audio"}

    text_meta = modalities["text"]
    audio_meta = modalities["audio"]
    assert text_meta["spans"] == audio_meta["spans"]
    assert len(text_meta["spans"]) == span_count

    expected_end_ms = int(round(audio_duration * 1000))
    assert text_meta["spans"][0][0] == 0
    assert text_meta["spans"][-1][-1] == expected_end_ms

    text_embeddings = np.asarray(text_meta["embeddings"], dtype=np.float32)
    audio_embeddings = np.asarray(audio_meta["embeddings"], dtype=np.float32)
    assert text_embeddings.shape == (span_count, 2)
    assert audio_embeddings.shape == (span_count, 4)
    assert not np.allclose(audio_embeddings, 0.0)

    expected_word_embs = [
        np.array([float(len(word)), float(sum(ord(ch) for ch in word) % 997)], dtype=np.float32)
        for word in transcript.split()
    ]
    np.testing.assert_allclose(text_embeddings[0], expected_word_embs[0])
    np.testing.assert_allclose(text_embeddings[-1], expected_word_embs[1])
    assert text_embeddings[:, 1].min() >= min(emb[1] for emb in expected_word_embs)
    assert text_embeddings[:, 1].max() <= max(emb[1] for emb in expected_word_embs)

    assert model.calls == transcript.split()

    provenance = publisher.kwargs["provenance"]
    assert set(provenance["modalities"]) == {"text", "audio"}
