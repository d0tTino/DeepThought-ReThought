import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.helpers import nats_server_available


@pytest.mark.nats
def test_multi_agent_demo_runs(tmp_path: Path) -> None:
    """The demo script should run and emit at least one agent message."""
    if not nats_server_available():
        pytest.skip("NATS server not available")

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(Path(__file__).resolve().parents[1] / "src"),
            str(Path(__file__).resolve().parents[1]),
        ]
    )

    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parents[1] / "examples" / "multi_agent_demo.py")],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=True,
        env=env,
    )

    assert "Agent 1 says" in result.stdout
