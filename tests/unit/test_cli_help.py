import os
import subprocess
import sys
from pathlib import Path


def test_cli_help(tmp_path):
    env = dict(**os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2] / "src"))
    result = subprocess.run(
        [sys.executable, "-m", "deepthought.cli", "--help"],
        stdout=subprocess.PIPE,
        text=True,
        check=True,
        env=env,
    )
    assert "usage" in result.stdout.lower()
