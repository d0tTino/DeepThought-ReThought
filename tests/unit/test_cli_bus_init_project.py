import os
import subprocess
import sys
from pathlib import Path


def test_bus_init_project(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
    subprocess.run(
        [sys.executable, "-m", "deepthought.cli", "bus", "init", "project", "demo"],
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        text=True,
        check=True,
        env=env,
    )
    dest = tmp_path / "demo"
    service_dir = dest / "src" / "deepthought" / "services" / "demo"
    assert dest.is_dir()
    docker_compose = dest / "docker-compose.yml"
    assert docker_compose.exists()
    assert service_dir.is_dir()
    assert (service_dir / "publisher.py").exists()
    assert (service_dir / "subscriber.py").exists()
    assert (service_dir / "service.py").exists()

    assert "deepthought_events" in docker_compose.read_text(encoding="utf-8")

    for py_file in service_dir.glob("*.py"):
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


def test_bus_init_project_options(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "deepthought.cli",
            "bus",
            "init",
            "project",
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
    service_dir = tmp_path / "opt" / "src" / "deepthought" / "services" / "opt"
    env_text = (service_dir / "nats.env.example").read_text(encoding="utf-8")
    assert "NATS_STREAM=custom" in env_text
    assert "NATS_TLS_CERT=c.pem" in env_text
    assert "NATS_JS_STORAGE=file" in env_text
    assert "NATS_MAX_MSGS=42" in env_text
    docker_text = (service_dir / "Dockerfile").read_text(encoding="utf-8")
    assert "pip install deepthought-rethought" in docker_text
    assert "NATS_TLS_CERT=c.pem" in docker_text
    assert "NATS_JS_STORAGE=file" in docker_text
    assert "NATS_MAX_MSGS=42" in docker_text

    compose_text = (tmp_path / "opt" / "docker-compose.yml").read_text(encoding="utf-8")
    assert "custom" in compose_text
    assert "c.pem" in compose_text
    assert "k.pem" in compose_text
    assert "ca.pem" in compose_text
