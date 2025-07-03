import os
import subprocess
import sys
from pathlib import Path


def _run_dtrt(tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2] / "src"))
    return subprocess.run(
        [sys.executable, "-m", "deepthought.cli", *args],
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )


def test_finetune_help(tmp_path: Path) -> None:
    result = _run_dtrt(tmp_path, "finetune", "--help")
    combined = result.stdout.lower() + result.stderr.lower()
    assert "usage" in combined


def test_init_service_demo(tmp_path: Path) -> None:
    result = _run_dtrt(tmp_path, "init", "service", "demo")
    assert result.returncode == 0
    service_dir = tmp_path / "src" / "deepthought" / "services" / "demo"
    assert service_dir.is_dir()
