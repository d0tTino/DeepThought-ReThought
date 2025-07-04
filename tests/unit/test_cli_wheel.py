import shutil
import subprocess
import sys
from pathlib import Path


def test_dtrt_init_service_from_wheel(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    wheel_dir = tmp_path / "dist"
    wheel_dir.mkdir()
    subprocess.run(
        [sys.executable, "setup.py", "bdist_wheel", "--dist-dir", str(wheel_dir)],
        cwd=repo_root,
        check=True,
    )

    try:
        wheel = next(wheel_dir.glob("*.whl"))
        venv_dir = tmp_path / "venv"
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
        pip = venv_dir / "bin" / "pip"
        dtrt = venv_dir / "bin" / "dtrt"
        subprocess.run([str(pip), "install", "--no-deps", str(wheel)], check=True)
        subprocess.run([str(dtrt), "init", "service", "demo"], cwd=tmp_path, check=True)
        assert (tmp_path / "src" / "deepthought" / "services" / "demo" / "service.py").exists()
    finally:
        shutil.rmtree(repo_root / "build", ignore_errors=True)
        shutil.rmtree(repo_root / "src" / "deepthought_rethought.egg-info", ignore_errors=True)
