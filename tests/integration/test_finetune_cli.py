import importlib
from pathlib import Path

import pytest

from deepthought.cli import main


def test_finetune_estimate_only(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("bitsandbytes")
    from transformers import AutoModelForCausalLM, GPT2Config

    train = importlib.import_module("deepthought.train")
    cfg = GPT2Config(n_embd=4, n_layer=1, n_head=1, vocab_size=10)
    dummy_model = AutoModelForCausalLM.from_config(cfg)
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", lambda *a, **k: dummy_model)
    monkeypatch.setattr(train, "run_training", lambda cfg: 0)

    dataset = Path(__file__).resolve().parents[1] / "data" / "finetune_sample.jsonl"
    out_dir = tmp_path / "model"
    rc = main(
        [
            "finetune",
            "--dataset-path",
            str(dataset),
            "--output-dir",
            str(out_dir),
            "--estimate-only",
        ]
    )
    assert rc == 0
    assert out_dir.is_dir()
