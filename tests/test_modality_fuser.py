import importlib
import pytest
import torch
import torch.nn.parameter as torch_parameter


@pytest.fixture(autouse=True)
def _ensure_real_torch():
    if not hasattr(torch_parameter.torch, "SymBool"):
        importlib.reload(torch_parameter)
        importlib.reload(torch.nn.modules.linear)

from deepthought.modules.fuser import ModalityFuser


def test_modality_fuser_basic():
    try:
        fuser = ModalityFuser({"text": 3, "audio": 1}, fused_dim=5)
    except AttributeError as exc:  # pragma: no cover - environment-specific
        pytest.skip(str(exc))
    text = torch.ones((2, 3))
    audio = torch.ones((2, 1))
    out = fuser({"text": text, "audio": audio})
    assert out.shape == (2, 5)


def test_modality_fuser_dropout(monkeypatch):
    try:
        fuser = ModalityFuser({"text": 2}, fused_dim=2, dropout_prob=1.0)
    except AttributeError as exc:  # pragma: no cover - environment-specific
        pytest.skip(str(exc))
    fuser.train()
    captured = {}
    original_forward = fuser.project.forward

    def capture(x):
        captured["input"] = x.detach().clone()
        return original_forward(x)

    monkeypatch.setattr(fuser.project, "forward", capture)
    text = torch.ones((3, 2))
    fuser({"text": text})
    assert torch.count_nonzero(captured["input"]) == 0


def test_modality_fuser_missing_modalities():
    try:
        fuser = ModalityFuser({"text": 2, "audio": 1}, fused_dim=3)
    except AttributeError as exc:  # pragma: no cover - environment-specific
        pytest.skip(str(exc))
    text = torch.ones((1, 2))
    with pytest.raises(RuntimeError):
        fuser({"text": text})
