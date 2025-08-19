import importlib
from unittest.mock import AsyncMock

import numpy as np
import pytest
import torch
import torch.nn.parameter as torch_parameter
from scipy.io import wavfile


@pytest.fixture(autouse=True)
def _ensure_real_torch():
    if not hasattr(torch_parameter.torch, "SymBool"):
        importlib.reload(torch_parameter)
        importlib.reload(torch.nn.modules.linear)


from deepthought.eda.events import EventSubjects, PerceptionEmbeddingsPayload
from deepthought.modules.fuser import ModalityFuser
from deepthought.services.perception.publisher import PerceptionPublisher
from deepthought.services.perception.user_embeddings import UserEmbeddings
from deepthought.services.perception.worker_audio import AudioPerceptionWorker
from deepthought.services.perception.worker_text import TextPerceptionWorker
from deepthought.services.perception.worker_video import VideoPerceptionWorker


class DummySentenceModel:
    def encode(self, text: str) -> np.ndarray:
        length = len(text)
        return np.asarray([length, length + 1], dtype=np.float32)


class DummyPublisher:
    def __init__(self, *args, **kwargs):
        self.publish = AsyncMock(return_value={"seq": 1})


@pytest.mark.asyncio
async def test_perception_pipeline_end_to_end(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "deepthought.services.perception.worker_text.SentenceTransformer",
        lambda name: DummySentenceModel(),
    )
    dummy_feats = np.array([[1.0, 2.0]], dtype=np.float32)
    dummy_times = np.array([0.0], dtype=np.float32)
    monkeypatch.setattr(
        "deepthought.services.perception.worker_video.video_to_feature_grid",
        lambda path, decode_fps, model_type, grid_fps: (dummy_feats, dummy_times),
    )
    monkeypatch.setattr(
        "deepthought.services.perception.publisher.Publisher",
        DummyPublisher,
    )

    sr = 16000
    audio_path = tmp_path / "a.wav"
    wavfile.write(audio_path, sr, np.ones(sr // 10, dtype=np.int16))
    audio_worker = AudioPerceptionWorker()
    audio_feats, _ = audio_worker(audio_path)

    text_worker = TextPerceptionWorker(hop_seconds=0.05)
    text_feats = text_worker([("hi", 0.0, 0.05)], tmp_path / "t.dat")

    video_worker = VideoPerceptionWorker()
    video_feats, _ = video_worker("v.mp4")

    store = UserEmbeddings(tmp_path / "emb.json")
    try:
        fuser = ModalityFuser({"audio": 1, "text": 2, "video": 2}, fused_dim=3, user_dim=2)
    except AttributeError as exc:  # pragma: no cover - environment-specific
        pytest.skip(str(exc))
    modalities = {
        "audio": torch.from_numpy(np.asarray(audio_feats[:1])).float(),
        "text": torch.from_numpy(np.asarray(text_feats[:1])).float(),
        "video": torch.from_numpy(np.asarray(video_feats[:1])).float(),
    }
    fused = fuser(modalities, user_embedding=torch.ones((1, 2)), user_id="u1", embedding_store=store)
    assert "u1" in store

    pub = PerceptionPublisher(nats_client=object(), js_context=object())
    await pub.publish(
        "m1",
        "u1",
        fused=fused.squeeze(0).detach().numpy().tolist(),
        by_modality={
            "text": {
                "spans": [[0, 1]],
                "embeddings": fused.squeeze(0).detach().numpy().reshape(1, -1).tolist(),
                "encoders": [],
            }
        },
    )
    pub._publisher.publish.assert_awaited_once()
    args, kwargs = pub._publisher.publish.call_args

    subject, payload = args
    assert subject == EventSubjects.PERCEPTION_EMBEDDINGS
    assert isinstance(payload, PerceptionEmbeddingsPayload)

