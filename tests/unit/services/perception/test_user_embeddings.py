import sys
import types
from pathlib import Path

import pytest

repo_root = Path(__file__).resolve().parents[4]

# Stub perception modules with heavy dependencies so we can import UserEmbeddings
services_pkg = types.ModuleType("deepthought.services")
services_pkg.__path__ = [str(repo_root / "src" / "deepthought" / "services")]
sys.modules["deepthought.services"] = services_pkg

perception_pkg = types.ModuleType("deepthought.services.perception")
perception_pkg.__path__ = [str(repo_root / "src" / "deepthought" / "services" / "perception")]
sys.modules["deepthought.services.perception"] = perception_pkg

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

# Stub prometheus_client to avoid requiring the dependency
prometheus_client = types.ModuleType("prometheus_client")
prometheus_client.Counter = lambda *a, **k: None
prometheus_client.Histogram = lambda *a, **k: None
prometheus_client.REGISTRY = types.SimpleNamespace(_names_to_collectors={})
sys.modules["prometheus_client"] = prometheus_client

# Stub deepthought.modules package to avoid heavy imports
modules_pkg = types.ModuleType("deepthought.modules")
modules_pkg.__path__ = [str(Path(__file__).resolve().parents[4] / "src" / "deepthought" / "modules")]
sys.modules["deepthought.modules"] = modules_pkg

try:  # Attempt to import the real torch library
    import torch  # noqa: F401
except Exception as exc:  # pragma: no cover - optional dependency
    pytest.skip(f"torch import failed: {exc}")

from deepthought.modules.fuser import ModalityFuser  # noqa: E402
from deepthought.services.perception.user_embeddings import UserEmbeddings  # noqa: E402


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


def test_update_from_gradient(tmp_path):
    path = tmp_path / "grad.json"
    store = UserEmbeddings(path)

    torch.manual_seed(0)
    modalities = {"text": torch.randn(1, 2)}
    fuser = ModalityFuser({"text": 2}, fused_dim=4, user_dim=3)

    emb = torch.zeros(1, 3, requires_grad=True)
    output = fuser(modalities, user_embedding=emb)
    loss = output.pow(2).sum()
    loss.backward()

    grad = emb.grad.squeeze(0)
    updated = store.update_from_gradient("carol", grad, lr=0.1)
    assert torch.allclose(updated, -0.1 * grad)


def test_update_from_bandit(tmp_path):
    path = tmp_path / "bandit.json"
    store = UserEmbeddings(path)

    torch.manual_seed(0)
    modalities = {"text": torch.randn(1, 2)}
    fuser = ModalityFuser({"text": 2}, fused_dim=4, user_dim=3)

    context = torch.tensor([0.5, -0.25, 0.1])
    reward = 2.0
    fuser.bandit_step(modalities, reward, context, user_id="dave", embedding_store=store)

    expected = 0.01 * reward * context
    stored = store.get("dave")
    assert stored is not None
    assert torch.allclose(stored, expected)
