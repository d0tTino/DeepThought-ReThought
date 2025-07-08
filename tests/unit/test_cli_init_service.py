import os
import subprocess
import sys
from pathlib import Path


def test_cli_init_service(tmp_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")

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
    assert (dest / "__init__.py").exists()
    assert (dest / "publisher.py").exists()
    assert (dest / "subscriber.py").exists()
    assert (dest / "Dockerfile").exists()

    class_name = "SampleService"
    pub_text = (dest / "publisher.py").read_text(encoding="utf-8")
    sub_text = (dest / "subscriber.py").read_text(encoding="utf-8")
    assert class_name + "Publisher" in pub_text
    assert class_name + "Subscriber" in sub_text

    import shutil

    shutil.rmtree(tmp_path, ignore_errors=True)
