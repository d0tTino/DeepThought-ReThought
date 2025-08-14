import torch

from deepthought.services.perception.user_embeddings import UserEmbeddings


def test_user_embeddings_persist_and_retrieve(tmp_path):
    path = tmp_path / "embeddings.json"
    store = UserEmbeddings(path)
    vec = torch.randn(4)
    store.set("alice", vec)

    reloaded = UserEmbeddings(path)
    fetched = reloaded.get("alice")
    assert fetched is not None
    assert torch.allclose(fetched, vec)
