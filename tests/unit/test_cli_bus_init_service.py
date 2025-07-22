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
    assert (dest / "service.py").exists()
    assert (dest / "Dockerfile").exists()

    for py_file in dest.glob("*.py"):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "py_compile",
                str(py_file),
            ],
            check=True,
            env=env,
        )


def test_bus_init_service_options(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "deepthought.cli",
            "bus",
            "init",
            "service",
            "opt",
            "--stream-name",
            "custom",
            "--tls-cert",
            "c.pem",
            "--tls-key",
            "k.pem",
            "--tls-ca",
            "ca.pem",
            "--js-storage",
            "file",
            "--max-msgs",
            "42",
        ],
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        text=True,
        check=True,
        env=env,
    )
    dest = tmp_path / "src" / "deepthought" / "services" / "opt"
    env_text = (dest / "nats.env.example").read_text(encoding="utf-8")
    assert "NATS_STREAM=custom" in env_text
    assert 'NATS_TLS_CERT="c.pem"' in env_text
    assert 'NATS_JS_STORAGE="file"' in env_text
    assert 'NATS_MAX_MSGS="42"' in env_text
    docker_text = (dest / "Dockerfile").read_text(encoding="utf-8")
    assert 'NATS_TLS_CERT="c.pem"' in docker_text
    assert 'NATS_JS_STORAGE="file"' in docker_text
    assert 'NATS_MAX_MSGS="42"' in docker_text
