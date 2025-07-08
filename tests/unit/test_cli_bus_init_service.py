import os
import subprocess
import sys
from pathlib import Path


def test_bus_init_service(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
    subprocess.run(
        [sys.executable, "-m", "deepthought.cli", "bus", "init", "service", "demo"],
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        text=True,
        check=True,
        env=env,
    )
    dest = tmp_path / "src" / "deepthought" / "services" / "demo"
    assert dest.is_dir()
    assert (dest / "__init__.py").exists()
    assert (dest / "publisher.py").exists()
    assert (dest / "subscriber.py").exists()
    assert (dest / "Dockerfile").exists()

    for py_file in dest.glob("*.py"):
        subprocess.run([
            sys.executable,
            "-m",
            "py_compile",
            str(py_file),
        ], check=True, env=env)
