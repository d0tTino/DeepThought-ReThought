import sys

import pytest

torch = sys.modules.get("torch")
if not getattr(torch, "nn", None):  # pragma: no cover - optional dependency missing
    pytest.skip("torch not available", allow_module_level=True)

from deepthought.modules.fuser import ModalityFuser
from deepthought.services.perception.user_embeddings import UserEmbeddings


def test_fuser_shape_with_user_embedding():
    fuser = ModalityFuser({"image": 4, "text": 6}, fused_dim=5, user_dim=3)
    modalities = {
        "image": torch.randn(2, 4),
        "text": torch.randn(2, 6),
    }
    user = torch.randn(2, 3)
    output = fuser(modalities, user)
    assert output.shape == (2, 5)


def test_fuser_modality_dropout_bias_only():
    fuser = ModalityFuser({"image": 2}, fused_dim=3, dropout_prob=1.0)
    fuser.train()
    modalities = {"image": torch.randn(1, 2)}
    result = fuser(modalities)
    expected = fuser.project.bias.unsqueeze(0)
    assert torch.allclose(result, expected)


def test_fuser_uses_store_when_available(tmp_path):
    store = UserEmbeddings(tmp_path / "store.json")
    store.set("u1", torch.ones(3))

    fuser = ModalityFuser({"text": 2}, fused_dim=4, user_dim=3)
    mods = {"text": torch.randn(1, 2)}
    out = fuser(mods, user_id="u1", embedding_store=store)
    assert out.shape == (1, 4)


def test_fuser_persists_user_embedding(tmp_path):
    store = UserEmbeddings(tmp_path / "store.json")
    fuser = ModalityFuser({"text": 2}, fused_dim=4, user_dim=3)
    mods = {"text": torch.randn(1, 2)}
    user_embed = torch.randn(1, 3)
    fuser(mods, user_embed, user_id="bob", embedding_store=store)

    retrieved = store.get("bob")
    assert retrieved is not None
    assert torch.allclose(retrieved, user_embed.squeeze(0))


def test_fuser_missing_modalities_raises():
    fuser = ModalityFuser({"text": 2, "audio": 1}, fused_dim=3)
    with pytest.raises(RuntimeError):
        fuser({"text": torch.randn(1, 2)})


def test_fuser_raises_on_mismatched_stored_vector(tmp_path):
    store = UserEmbeddings(tmp_path / "store.json")
    store.set("u1", torch.ones(4))

    fuser = ModalityFuser({"text": 2}, fused_dim=4, user_dim=3)
    mods = {"text": torch.randn(1, 2)}
    with pytest.raises(ValueError, match="dimension"):
        fuser(mods, user_id="u1", embedding_store=store)


def test_fuser_raises_on_mismatched_provided_vector(tmp_path):
    store = UserEmbeddings(tmp_path / "store.json")
    fuser = ModalityFuser({"text": 2}, fused_dim=4, user_dim=3)
    mods = {"text": torch.randn(1, 2)}
    bad_embed = torch.randn(1, 4)
    with pytest.raises(ValueError, match="dimension"):
        fuser(mods, bad_embed, user_id="u1", embedding_store=store)
