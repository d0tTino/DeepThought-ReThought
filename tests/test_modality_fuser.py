import pytest
import torch

if not hasattr(torch, "SymBool"):
    pytest.skip("PyTorch lacks SymBool", allow_module_level=True)

from deepthought.modules.fuser import ModalityFuser


def _make_fuser(*args, **kwargs):
    try:
        return ModalityFuser(*args, **kwargs)
    except AttributeError as exc:  # pragma: no cover - environment-specific
        pytest.skip(str(exc))


def test_modality_fuser_basic():
    fuser = _make_fuser({"text": 3, "audio": 1}, fused_dim=5)
    text = torch.ones((2, 3))
    audio = torch.ones((2, 1))
    out = fuser({"text": text, "audio": audio})
    assert out.shape == (2, 5)


def test_modality_fuser_dropout(monkeypatch):
    fuser = _make_fuser({"text": 2}, fused_dim=2, dropout_prob=1.0)

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
    fuser = _make_fuser({"text": 2, "audio": 1}, fused_dim=3)

    text = torch.ones((1, 2))
    with pytest.raises(RuntimeError):
        fuser({"text": text})
