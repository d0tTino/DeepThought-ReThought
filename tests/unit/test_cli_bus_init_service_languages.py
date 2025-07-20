import os
import subprocess
import sys
from pathlib import Path


def _run(tmp_path: Path, *args: str) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
    subprocess.run(
        [sys.executable, "-m", "deepthought.cli", *args],
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        text=True,
        check=True,
        env=env,
    )


def test_bus_init_service_go(tmp_path: Path) -> None:
    _run(tmp_path, "bus", "init", "service", "gosvc", "--language", "go")
    dest = tmp_path / "src" / "deepthought" / "services" / "gosvc"
    assert (dest / "main.go").exists()
    assert (dest / "go.mod").exists()


def test_bus_init_service_ts(tmp_path: Path) -> None:
    _run(tmp_path, "bus", "init", "service", "tssvc", "--language", "ts")
    dest = tmp_path / "src" / "deepthought" / "services" / "tssvc"
    assert (dest / "src" / "index.ts").exists()
    assert (dest / "package.json").exists()
