import importlib
import sys

sys.modules.pop("torch", None)
torch = importlib.import_module("torch")

from deepthought.modules.fuser import ModalityFuser


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
