# ruff: noqa: E402
from __future__ import annotations

import os
import sys
import types
import json
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest


os.environ.setdefault("DEEPTHOUGHT_LIGHT_IMPORT", "1")

modules_pkg = types.ModuleType("deepthought.modules")
modules_pkg.__path__ = ["src/deepthought/modules"]
modules_pkg.ModalityFuser = object  # type: ignore[attr-defined]
sys.modules.setdefault("deepthought.modules", modules_pkg)

services_pkg = types.ModuleType("deepthought.services")
services_pkg.__path__ = ["src/deepthought/services"]
sys.modules.setdefault("deepthought.services", services_pkg)

perception_pkg = types.ModuleType("deepthought.services.perception")
perception_pkg.__path__ = ["src/deepthought/services/perception"]
sys.modules.setdefault("deepthought.services.perception", perception_pkg)
services_pkg.perception = perception_pkg  # type: ignore[attr-defined]

worker_text_stub = types.ModuleType("deepthought.services.perception.worker_text")
worker_text_stub.TextPerceptionWorker = object  # type: ignore[attr-defined]
worker_text_stub.Token = tuple  # type: ignore[attr-defined]
sys.modules.setdefault("deepthought.services.perception.worker_text", worker_text_stub)

worker_audio_stub = types.ModuleType("deepthought.services.perception.worker_audio")
worker_audio_stub.AudioPerceptionWorker = object  # type: ignore[attr-defined]
sys.modules.setdefault("deepthought.services.perception.worker_audio", worker_audio_stub)

worker_video_stub = types.ModuleType("deepthought.services.perception.worker_video")
worker_video_stub.VideoPerceptionWorker = object  # type: ignore[attr-defined]
sys.modules.setdefault("deepthought.services.perception.worker_video", worker_video_stub)

publisher_stub = types.ModuleType("deepthought.services.perception.publisher")
class _BasePublisher:
    async def publish(self, **kwargs):
        raise NotImplementedError
publisher_stub.PerceptionPublisher = _BasePublisher  # type: ignore[attr-defined]
sys.modules.setdefault("deepthought.services.perception.publisher", publisher_stub)

from deepthought.eda.events import EventSubjects, PerceptionEmbeddingsEvent
from deepthought.services.perception.ingestion_worker import AttachmentIngestionWorker
import deepthought.services.perception.listener as listener_mod
import deepthought.services.perception.service as service_mod
from deepthought.services.perception.listener import PerceptionServiceListener
from deepthought.services.perception.service import PerceptionService
import deepthought.services.perception_interpret_service as interpret_mod
from deepthought.services.perception_interpret_service import PerceptionInterpretService


class _Msg:
    def __init__(self, payload: dict):
        self.data = json.dumps(payload).encode()
        self.acked = False
        self.nacked = False

    async def ack(self):
        self.acked = True

    async def nak(self):
        self.nacked = True


class _DummySubscriber:
    def __init__(self, *_args, **_kwargs):
        pass

    async def subscribe(self, **_kwargs):
        return True

    async def unsubscribe_all(self):
        return None


class _RecordingPublisher:
    def __init__(self, *_args, **_kwargs):
        self.published = []

    async def publish(self, subject, payload, **kwargs):
        self.published.append((subject, payload, kwargs))


class _RecordingPerceptionPublisher:
    def __init__(self):
        self.embeddings = []
        self.modality_results = []

    async def publish(self, **kwargs):
        self.embeddings.append(kwargs)

    async def publish_modality_result(self, **kwargs):
        self.modality_results.append(kwargs)


class _TextWorker:
    def __call__(self, tokens, memmap_path):
        data = np.array([[1.0, 2.0]], dtype="float32")
        mm = np.memmap(memmap_path, dtype="float32", mode="w+", shape=data.shape)
        mm[:] = data
        mm.flush()
        times = np.array([[0.0, 0.05]], dtype=np.float32)
        return mm, times


class _AudioWorker:
    cache_dir = None
    model = "a"
    window_size = 0.05
    step_size = 0.05
    model_path = None

    def __call__(self, path, cache_dir=None):
        return np.array([[0.1, 0.2]], dtype="float32"), np.array([[0.0, 0.05]], dtype=np.float32)




class _Fuser:
    modality_dims = {"text": 2, "audio": 2, "video": 2}
    user_dim = 0

    def __call__(self, aligned_modalities, **_kwargs):
        import torch

        stacks = [tensor for tensor in aligned_modalities.values()]
        return torch.stack(stacks).mean(dim=0)

class _VideoWorker:
    cache_dir = None
    decode_fps = 1
    model_type = "v"
    grid_fps = 1

    def __call__(self, path):
        return np.array([[0.3, 0.4]], dtype="float32"), np.array([[0.0, 1.0]], dtype=np.float32)


@pytest.mark.asyncio
async def test_attachment_ingestion_to_embedding_to_interpretation(monkeypatch, tmp_path):
    monkeypatch.setattr(listener_mod, "Subscriber", _DummySubscriber)
    monkeypatch.setattr(interpret_mod, "Subscriber", _DummySubscriber)
    monkeypatch.setattr(interpret_mod, "Publisher", _RecordingPublisher)
    monkeypatch.setattr(
        service_mod, "get_settings",
        lambda: type("S", (), {"wandb_enabled": False, "perception_inference_timeout_seconds": 10, "perception_worker_retries": 1})(),
    )
    monkeypatch.setattr(
        service_mod, "PerceptionConfig",
        lambda: type(
            "C",
            (),
            {
                "grid_hop_size": None,
                "text_cache_dir": None,
                "audio_cache_dir": str(tmp_path),
                "video_cache_dir": str(tmp_path),
                "audio_window_size": 0.05,
                "audio_hop_size": 0.05,
                "audio_model": "a",
                "audio_model_path": None,
                "text_model": "t",
                "text_hop_size": 0.05,
                "video_model": "v",
                "video_hop_size": 1.0,
                "modality_dims": {"text": 2, "audio": 2, "video": 2},
                "enable_asr_transcription": False,
            },
        )(),
    )

    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    (fixtures / "sample.png").write_bytes(b"\x89PNG\r\n\x1a\nfixture")
    (fixtures / "sample.wav").write_bytes(b"RIFFfixtureWAVE")
    (fixtures / "sample.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42fixture")

    handler = partial(SimpleHTTPRequestHandler, directory=str(fixtures))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"

    perception_publisher = _RecordingPerceptionPublisher()
    service = PerceptionService(
        perception_publisher,
        text_worker=_TextWorker(),
        audio_worker=_AudioWorker(),
        video_worker=_VideoWorker(),
        fuser=_Fuser(),
    )
    listener = PerceptionServiceListener(
        service,
        object(),
        object(),
        ingestion_worker=AttachmentIngestionWorker(allowed_schemes=("http", "https"), retries=1),
    )

    extract_msg = _Msg(
        {
            "message_id": "m1",
            "user_id": "u1",
            "input_id": "in-1",
            "audio_opt_in": True,
            "video_opt_in": True,
            "attachments": [
                {"url": f"{base_url}/sample.png", "content_type": "image/png", "filename": "sample.png"},
                {"url": f"{base_url}/sample.wav", "content_type": "audio/wav", "filename": "sample.wav"},
                {"url": f"{base_url}/sample.mp4", "content_type": "video/mp4", "filename": "sample.mp4"},
            ],
            "text": "hello",
        }
    )
    await listener.handle_extract(extract_msg)
    server.shutdown()
    thread.join(timeout=2)

    assert extract_msg.acked is True
    assert len(perception_publisher.embeddings) == 1
    run_payload = perception_publisher.embeddings[0]
    assert run_payload["input_id"] == "in-1"
    assert set(run_payload["by_modality"]) == {"text", "audio", "video"}
    assert {m["modality"] for m in perception_publisher.modality_results if m["success"]} >= {"text", "audio", "video"}

    interpret_service = PerceptionInterpretService(object(), object())
    evt = PerceptionEmbeddingsEvent.from_dict(
        {
            "payload": {
                "message_id": "m1",
                "user_id": "u1",
                "input_id": "in-1",
                "confidence": 0.8,
                "modality_confidence": {"image": 0.8, "audio": 0.7, "video": 0.9},
                "by_modality": run_payload["by_modality"],
            }
        }
    )
    emb_msg = _Msg(json.loads(evt.to_json()))
    await interpret_service._handle_embeddings(emb_msg)
    req_msg = _Msg({"input_id": "in-1", "attachments": [{"url": "https://x/a.png", "content_type": "image/png"}]})
    await interpret_service._handle_interpret_request(req_msg)

    assert emb_msg.acked is True
    assert req_msg.acked is True
    assert interpret_service._publisher.published
    assert interpret_service._publisher.published[0][0] == EventSubjects.PERCEPTION_INTERPRET_RETRIEVED
