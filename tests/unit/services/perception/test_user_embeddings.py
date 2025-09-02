import sys
import types

import pytest

# Stub perception modules with heavy dependencies so we can import UserEmbeddings
for name, attr in [
    ("cli", "main"),
    ("config", "PerceptionConfig"),
    ("publisher", "PerceptionPublisher"),
    ("service", "PerceptionService"),
    ("worker_audio", "AudioPerceptionWorker"),
    ("worker_text", "TextPerceptionWorker"),
    ("worker_video", "VideoPerceptionWorker"),
]:
    mod = types.ModuleType(f"deepthought.services.perception.{name}")
    if attr == "main":
        setattr(mod, attr, lambda *a, **k: None)
    else:
        setattr(mod, attr, type(attr, (), {}))
    sys.modules[f"deepthought.services.perception.{name}"] = mod

try:  # Attempt to import the real torch library
    import torch  # noqa: F401
except Exception as exc:  # pragma: no cover - optional dependency
    pytest.skip(f"torch import failed: {exc}")

from deepthought.modules.fuser import ModalityFuser
from deepthought.services.perception.user_embeddings import UserEmbeddings


def test_modality_fuser_updates_and_retrieves(tmp_path):
    path = tmp_path / "embeddings.json"
    store = UserEmbeddings(path)

    torch.manual_seed(0)
    fuser = ModalityFuser({"text": 2}, fused_dim=4, user_dim=3)

    modalities = {"text": torch.randn(1, 2)}
    emb1 = torch.randn(1, 3)
    fuser(modalities, user_embedding=emb1, user_id="bob", embedding_store=store)
    stored = store.get("bob")
    assert stored is not None
    assert torch.allclose(stored, emb1.squeeze(0))

    emb2 = torch.randn(1, 3)
    fuser(modalities, user_embedding=emb2, user_id="bob", embedding_store=store)
    stored2 = store.get("bob")
    assert stored2 is not None
    assert torch.allclose(stored2, emb2.squeeze(0))

    manual = fuser(modalities, user_embedding=stored2.unsqueeze(0))
    retrieved = fuser(modalities, user_id="bob", embedding_store=store)
    assert torch.allclose(retrieved, manual)


def test_user_embedding_persistence_across_instances(tmp_path):
    path = tmp_path / "store.json"
    store = UserEmbeddings(path)

    torch.manual_seed(0)
    modalities = {"text": torch.randn(1, 2)}
    fuser = ModalityFuser({"text": 2}, fused_dim=4, user_dim=3)

    emb1 = torch.randn(1, 3)
    fuser(modalities, user_embedding=emb1, user_id="alice", embedding_store=store)
    emb2 = torch.randn(1, 3)
    fuser(modalities, user_embedding=emb2, user_id="alice", embedding_store=store)

    new_store = UserEmbeddings(path)
    new_fuser = ModalityFuser({"text": 2}, fused_dim=4, user_dim=3)
    manual = new_fuser(modalities, user_embedding=emb2)
    retrieved = new_fuser(modalities, user_id="alice", embedding_store=new_store)
    assert torch.allclose(retrieved, manual)

    stored = new_store.get("alice")
    assert stored is not None
    assert torch.allclose(stored, emb2.squeeze(0))
