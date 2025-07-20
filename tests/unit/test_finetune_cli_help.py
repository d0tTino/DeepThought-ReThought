import os
import subprocess
import sys
from pathlib import Path


def test_finetune_cli_help(tmp_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
    result = subprocess.run(
        [sys.executable, "-m", "deepthought.cli", "finetune", "--help"],
        stdout=subprocess.PIPE,
        text=True,
        check=True,
        env=env,
    )
    assert "usage" in result.stdout.lower()
    assert "--estimate-vram" in result.stdout
    assert "--estimate-only" in result.stdout
    assert "--pack-sequences" in result.stdout
    assert "--epochs" in result.stdout
    assert "--batch-size" in result.stdout
    assert "--lr" in result.stdout
    assert "--model-loader" in result.stdout
    assert "--dataset-loader" in result.stdout
