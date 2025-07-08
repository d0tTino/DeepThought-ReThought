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
    assert (service_dir / "__init__.py").exists()


def test_init_service_demo_start_stop(tmp_path: Path) -> None:
    result = _run_dtrt(tmp_path, "init", "service", "demo")
    assert result.returncode == 0
    service_dir = tmp_path / "src" / "deepthought" / "services" / "demo"
    assert (service_dir / "__init__.py").exists()
    svc_file = service_dir / "service.py"

    import importlib.util

    if svc_file.exists():
        spec = importlib.util.spec_from_file_location("demo.service", svc_file)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        DemoService = module.DemoService
    else:
        fallback = Path(__file__).resolve().parents[2] / "tools" / "template_service" / "service.py"
        spec = importlib.util.spec_from_file_location("template_service.service", fallback)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        DemoService = module.TemplateService

    class DummyNATS:
        def __init__(self) -> None:
            self.is_connected = True

    class DummyJS:
        async def publish(self, *a, **k):
            class Ack:
                seq = 1
                stream = "s"

            return Ack()

        async def subscribe(self, *a, **k):
            class Sub:
                async def unsubscribe(self) -> None:
                    pass

            return Sub()

    async def main() -> None:
        svc = DemoService(DummyNATS(), DummyJS())
        await svc.start()
        await svc.stop()

    import asyncio

    asyncio.run(main())


from deepthought.cli import _build_parser


def test_parse_finetune_args():
    parser = _build_parser()
    args = parser.parse_args(
        [
            "finetune",
            "--model-path",
            "mp",
            "--dataset-path",
            "ds",
            "--bits",
            "8",
            "--output-dir",
            "/tmp/out",
            "--max-seq-length",
            "4096",
            "--pack-sequences",
            "--estimate-vram",
        ]
    )
    assert args.command == "finetune"
    assert args.model_path == "mp"
    assert args.dataset_path == "ds"
    assert args.bits == 8
    assert args.output_dir == "/tmp/out"
    assert args.max_seq_length == 4096
    assert args.pack_sequences
    assert args.estimate_vram
    assert args.func.__name__ == "_cmd_finetune"


def test_parse_bus_init_service():
    parser = _build_parser()
    args = parser.parse_args(["bus", "init", "service", "foo"])
    assert args.command == "bus"
    assert args.bus_cmd == "init"
    assert args.target == "service"
    assert args.name == "foo"
    assert args.template == "bus_service"
    assert args.func.__name__ == "_cmd_init_service"
