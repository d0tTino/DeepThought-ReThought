import os
import subprocess
import sys
from pathlib import Path


def test_cli_init_service(tmp_path):
    env = dict(**os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2] / "src"))
    service_name = "sample"
    subprocess.run(
        [sys.executable, "-m", "deepthought.cli", "init", "service", service_name],
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        text=True,
        check=True,
        env=env,
    )
    dest = tmp_path / "src" / "deepthought" / "services" / service_name
    assert dest.is_dir()
    assert (dest / "publisher.py").exists()
    assert (dest / "subscriber.py").exists()
    assert (dest / "Dockerfile").exists()
