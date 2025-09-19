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


MODULE_PATH = Path(__file__).with_name("fuser_train.py")
PROJECT_ROOT = MODULE_PATH.resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
SPEC = importlib.util.spec_from_file_location("fuser_train", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
fuser_train = importlib.util.module_from_spec(SPEC)
sys.modules["fuser_train"] = fuser_train
SPEC.loader.exec_module(fuser_train)

main = fuser_train.main
parse_args = fuser_train.parse_args


def test_parse_args_basic():
    args = parse_args(["--features", "feats.npz", "--output", "model.pt"])
    assert args.features == Path("feats.npz")
    assert args.output == Path("model.pt")
    assert args.epochs == 5
    assert args.shuffle is True


def _wandb_stub():
    wandb_module = types.ModuleType("wandb")
    wandb_module.logged: list[tuple[dict[str, float], dict[str, int]]] = []
    wandb_module.inits: list[dict[str, object]] = []
    wandb_module.artifacts: list[object] = []
    wandb_module.finished = 0
    wandb_module.summary = {}

    class Artifact:
        def __init__(self, name: str, type: str, metadata: dict | None = None) -> None:
            self.name = name
            self.type = type
            self.metadata = metadata or {}
            self.files: list[Path] = []

        def add_file(self, filename: str) -> None:  # pragma: no cover - simple helper
            self.files.append(Path(filename))

    def init(**kwargs):  # pragma: no cover - simple stub
        wandb_module.inits.append(kwargs)

        class Run:
            def finish(self) -> None:  # pragma: no cover - simple helper
                wandb_module.finished += 1

        return Run()

    def log(data, **kwargs):
        wandb_module.logged.append((data, kwargs))

    def log_artifact(artifact):  # pragma: no cover - simple stub
        wandb_module.artifacts.append(artifact)

    def finish():  # pragma: no cover - simple stub
        wandb_module.finished += 1

    wandb_module.init = init
    wandb_module.log = log
    wandb_module.log_artifact = log_artifact
    wandb_module.finish = finish
    wandb_module.Artifact = Artifact
    return wandb_module


def test_main_trains_and_logs_with_user_embeddings(tmp_path, monkeypatch):
    feats = tmp_path / "feats.npz"
    num_samples = 6
    np.savez(
        feats,
        vision=np.ones((num_samples, 2), dtype=np.float32),
        audio=np.ones((num_samples, 3), dtype=np.float32),
        user_embeddings=np.full((num_samples, 4), 0.5, dtype=np.float32),
        user_ids=np.array([f"user-{i}" for i in range(num_samples)]),
        target=np.zeros((num_samples, 5), dtype=np.float32),
    )

    wandb_module = _wandb_stub()
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
        "3",
        "--lr",
        "1e-2",
        "--dropout-prob",
        "0.1",
        "--wandb-project",
        "demo",
        "--wandb-run-name",
        "unit-test",
    ]
    assert main(argv) == 0
    assert out_path.exists()
    assert wandb_module.logged, "W&B logging did not occur"
    assert wandb_module.artifacts, "Model artifact was not logged"
    artifact = wandb_module.artifacts[0]
    assert artifact.files and out_path in artifact.files
    assert artifact.metadata["modalities"] == ["audio", "vision"]
    assert wandb_module.inits[0]["config"]["use_user_embeddings"] is True
    assert wandb_module.finished == 1


def test_main_runs_without_user_embeddings(tmp_path, monkeypatch):
    feats = tmp_path / "feats.npz"
    num_samples = 4
    np.savez(
        feats,
        vision=np.ones((num_samples, 2), dtype=np.float32),
        audio=np.ones((num_samples, 3), dtype=np.float32),
        target=np.zeros((num_samples, 5), dtype=np.float32),
    )

    wandb_module = _wandb_stub()
    monkeypatch.setitem(sys.modules, "wandb", wandb_module)

    out_path = tmp_path / "model.pt"
    argv = [
        "--features",
        str(feats),
        "--output",
        str(out_path),
        "--epochs",
        "1",
        "--batch-size",
        "2",
        "--wandb-project",
        "demo",
    ]
    assert main(argv) == 0
    assert out_path.exists()
    assert wandb_module.logged
    config = wandb_module.inits[0]["config"]
    assert config["use_user_embeddings"] is False
    assert config["user_embedding_dim"] == 0
    assert wandb_module.artifacts

