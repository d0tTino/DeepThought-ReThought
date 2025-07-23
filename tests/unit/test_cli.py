import os
import subprocess
import sys
from pathlib import Path

import pytest


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
    import sys
    import types

    fake_nats = types.ModuleType("nats")
    import importlib.machinery

    fake_nats.__spec__ = importlib.machinery.ModuleSpec("nats", loader=None)
    fake_aio = types.ModuleType("aio")
    fake_client = types.ModuleType("client")
    setattr(fake_client, "Client", object)
    fake_msg_mod = types.ModuleType("msg")
    setattr(fake_msg_mod, "Msg", object)
    fake_aio.client = fake_client
    fake_aio.msg = fake_msg_mod
    fake_js = types.ModuleType("js")
    fake_js_client = types.ModuleType("client")
    setattr(fake_js_client, "JetStreamContext", object)
    fake_js.client = fake_js_client
    fake_nats.aio = fake_aio
    fake_nats.js = fake_js
    sys.modules.setdefault("nats", fake_nats)
    sys.modules.setdefault("nats.aio", fake_aio)
    sys.modules.setdefault("nats.aio.client", fake_client)
    sys.modules.setdefault("nats.aio.msg", fake_msg_mod)
    sys.modules.setdefault("nats.js", fake_js)
    sys.modules.setdefault("nats.js.client", fake_js_client)

    if svc_file.exists():
        spec = importlib.util.spec_from_file_location("demo.service", svc_file)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        DemoService = module.DemoService
    else:
        from deepthought.templates.bus_service import service as module

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
            "auto",
            "--epochs",
            "2",
            "--batch-size",
            "4",
            "--lr",
            "0.001",
            "--estimate-vram",
        ]
    )
    assert args.command == "finetune"
    assert args.model_path == "mp"
    assert args.dataset_path == "ds"
    assert args.bits == 8
    assert args.output_dir == "/tmp/out"
    assert args.max_seq_length == 4096
    assert args.pack_sequences == "auto"
    assert args.epochs == 2
    assert args.batch_size == 4
    assert args.lr == 0.001
    assert args.estimate_vram
    assert args.model_loader == "hf"
    assert args.dataset_loader == "hf"
    assert args.func.__name__ == "_cmd_finetune"


def test_parse_bus_init_service():
    parser = _build_parser()
    args = parser.parse_args(
        [
            "bus",
            "init",
            "service",
            "foo",
            "--stream-name",
            "bar",
            "--tls-cert",
            "c.pem",
            "--tls-key",
            "k.pem",
            "--tls-ca",
            "ca.pem",
            "--js-storage",
            "file",
            "--max-msgs",
            "123",
            "--language",
            "go",
        ]
    )
    assert args.command == "bus"
    assert args.bus_cmd == "init"
    assert args.target == "service"
    assert args.name == "foo"
    assert args.template == "bus_service"
    assert args.func.__name__ == "_cmd_init_service"
    assert args.stream_name == "bar"
    assert args.tls_cert == "c.pem"
    assert args.tls_key == "k.pem"
    assert args.tls_ca == "ca.pem"
    assert args.js_storage == "file"
    assert args.max_msgs == 123
    assert args.language == "go"


def test_finetune_estimate_vram(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("bitsandbytes")
    import importlib

    from transformers import AutoModelForCausalLM, GPT2Config

    train = importlib.import_module("deepthought.train")
    cfg = GPT2Config(n_embd=4, n_layer=1, n_head=1, vocab_size=10)
    dummy_model = AutoModelForCausalLM.from_config(cfg)
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", lambda *a, **k: dummy_model)
    monkeypatch.setattr(train, "run_training", lambda cfg: 0)

    result = _run_dtrt(
        tmp_path,
        "finetune",
        "--dataset-path",
        "ds",
        "--estimate-vram",
    )
    assert "Estimated VRAM requirement" in result.stdout
