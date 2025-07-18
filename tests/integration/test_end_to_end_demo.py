import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.slow
def test_end_to_end_demo_runs(tmp_path: Path) -> None:
    if shutil.which("nats-server") is None:
        pytest.skip("nats-server executable not found")

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(Path(__file__).resolve().parents[1] / "src"),
            str(Path(__file__).resolve().parents[1]),
        ]
    )
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parents[1] / "examples" / "end_to_end_demo.py")],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=True,
        env=env,
    )
    assert "End-to-end demo complete" in result.stdout
