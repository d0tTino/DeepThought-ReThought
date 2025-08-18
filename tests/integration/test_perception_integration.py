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
from deepthought.services.perception.publisher import PerceptionPublisher
from deepthought.services.perception.service import PerceptionService


class DummyTextWorker:
    def __call__(self, tokens, memmap_path):
        data = np.array([[1.0, 2.0]], dtype="float32")
        mm = np.memmap(memmap_path, dtype="float32", mode="w+", shape=data.shape)
        mm[:] = data
        mm.flush()
        return mm


class DummyAudioWorker:
    def __call__(self, audio_path):
        feats = np.array([[0.5, 1.5]], dtype="float32")
        times = np.array([0.0], dtype="float32")
        return feats, times


class DummyPublisher:
    def __init__(self, *args, **kwargs):
        self.publish = AsyncMock(return_value={"seq": 1})


@pytest.mark.asyncio
async def test_service_end_to_end(monkeypatch):
    monkeypatch.setattr(
        "deepthought.services.perception.publisher.Publisher",
        DummyPublisher,
    )

    publisher = PerceptionPublisher(nats_client=object(), js_context=object())
    service = PerceptionService(
        publisher,
        text_worker=DummyTextWorker(),
        audio_worker=DummyAudioWorker(),
    )

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
    assert payload.embeddings == [[1.0, 2.0], [0.5, 1.5]]
    assert payload.spans == [[0, 1], [1, 2]]
    assert payload.encoders == [
        {"name": "DummyTextWorker"},
        {"name": "DummyAudioWorker"},
    ]
    assert payload.provenance == {
        "source": "integration",
        "modalities": ["text", "audio"],
    }
