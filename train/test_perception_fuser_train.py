import importlib
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

try:  # pragma: no cover - optional dependency
    torch = importlib.import_module("torch")  # noqa: F401
except Exception as exc:  # pragma: no cover - environment without torch
    pytest.skip(str(exc), allow_module_level=True)


MODULE_PATH = Path(__file__).with_name("perception_fuser_train.py")
PROJECT_ROOT = MODULE_PATH.resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
SPEC = importlib.util.spec_from_file_location(
    "perception_fuser_train", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
perception_fuser_train = importlib.util.module_from_spec(SPEC)
sys.modules["perception_fuser_train"] = perception_fuser_train
SPEC.loader.exec_module(perception_fuser_train)

main = perception_fuser_train.main
parse_args = perception_fuser_train.parse_args


def test_parse_args_basic():
    args = parse_args(["--features", "feats.npz", "--output", "model.pt"])
    assert args.features == Path("feats.npz")
    assert args.output == Path("model.pt")
    assert args.epochs == 5
    assert args.shuffle is True


def test_main_trains_and_logs(tmp_path, monkeypatch):
    feats = tmp_path / "feats.npz"
    np.savez(
        feats,
        text=np.ones((4, 2), dtype=np.float32),
        audio=np.ones((4, 3), dtype=np.float32),
        target=np.zeros((4, 5), dtype=np.float32),
    )

    wandb_module = types.ModuleType("wandb")
    wandb_module.logged: list[tuple[dict[str, float], dict[str, int]]] = []
    wandb_module.inits: list[dict[str, object]] = []
    wandb_module.finished = 0

    def init(**kwargs):  # pragma: no cover - simple stub
        wandb_module.inits.append(kwargs)
        return wandb_module

    def log(data, **kwargs):
        wandb_module.logged.append((data, kwargs))

    def finish():  # pragma: no cover - simple stub
        wandb_module.finished += 1

    wandb_module.init = init
    wandb_module.log = log
    wandb_module.finish = finish

    monkeypatch.setitem(sys.modules, "wandb", wandb_module)

    out_path = tmp_path / "model.pt"
    argv = [
        "--features",
        str(feats),
        "--output",
        str(out_path),
        "--epochs",
        "2",
        "--batch-size",
        "2",
        "--wandb-project",
        "demo",
        "--wandb-run-name",
        "unit-test",
    ]
    assert main(argv) == 0
    assert out_path.exists()
    assert wandb_module.logged, "W&B logging did not occur"
    assert wandb_module.inits[0]["project"] == "demo"
    assert wandb_module.finished == 1
