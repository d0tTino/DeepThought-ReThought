import os
import shutil
from pathlib import Path

from deepthought.cli import main


def test_cli_bus_init(tmp_path: Path) -> None:
    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        main(["bus", "init", "service", "demo"])
    finally:
        os.chdir(cwd)
    svc_file = tmp_path / "src" / "deepthought" / "services" / "demo" / "service.py"
    assert svc_file.exists()
    text = svc_file.read_text(encoding="utf-8")
    assert "class DemoService" in text
    shutil.rmtree(tmp_path / "src", ignore_errors=True)
