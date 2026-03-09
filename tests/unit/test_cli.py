import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from deepthought.cli import _build_parser


def _run_dtrt(
    tmp_path: Path, *args: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess:
    run_env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2] / "src"))
    if env:
        run_env.update(env)
    return subprocess.run(
        [sys.executable, "-m", "deepthought.cli", *args],
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=run_env,
    )


def test_finetune_help(tmp_path: Path) -> None:
    result = _run_dtrt(tmp_path, "finetune", "--help")
    combined = result.stdout.lower() + result.stderr.lower()
    assert "usage" in combined


def test_finetune_estimate_only(tmp_path: Path) -> None:
    shim_dir = tmp_path / "shim"
    shim_dir.mkdir()
    shim_path = shim_dir / "sitecustomize.py"
    shim_path.write_text(
        textwrap.dedent(
            """
            from dataclasses import dataclass
            import sys
            import types


            @dataclass
            class TrainingConfig:
                model_path: str
                dataset_path: str
                model_loader: str
                dataset_loader: str
                bits: int
                output_dir: str
                max_seq_length: int
                pack_sequences: str
                epochs: float
                batch_size: int
                lr: float
                resume: bool
                lora_r: int
                lora_alpha: int
                lora_dropout: float
                lora_target_modules: tuple[str, ...]
                use_nf4: bool
                use_double_quant: bool
                compute_dtype: str


            def load_model(*_args, **_kwargs):
                return object(), None


            def estimate_vram(*_args, **_kwargs):
                return 1.23


            def run_training(*_args, **_kwargs):
                return 0


            module = types.ModuleType("deepthought.train")
            module.TrainingConfig = TrainingConfig
            module.load_model = load_model
            module.estimate_vram = estimate_vram
            module.run_training = run_training
            sys.modules["deepthought.train"] = module
            """
        )
    )

    src_path = Path(__file__).resolve().parents[2] / "src"
    env = {"PYTHONPATH": f"{shim_dir}:{src_path}"}

    result = _run_dtrt(
        tmp_path,
        "finetune",
        "--estimate-only",
        "--model-path",
        "gpt2",
        "--max-seq-length",
        "8",
        "--batch-size",
        "1",
        env=env,
    )

    assert result.returncode == 0
    assert "Estimated VRAM requirement: 1.23 GB" in result.stdout


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


def test_parse_perception_run():
    parser = _build_parser()
    args = parser.parse_args(["perception", "run", "--message-id", "m1", "--user-id", "u1"])
    assert args.command == "perception"
    assert args.perception_cmd == "run"
    assert args.message_id == "m1"
    assert args.user_id == "u1"
    assert args.func.__name__ == "_cmd_perception_run"


def test_parse_perception_delete_user():
    parser = _build_parser()
    args = parser.parse_args(["perception", "delete-user", "--user-id", "u1"])
    assert args.command == "perception"
    assert args.perception_cmd == "delete-user"
    assert args.user_id == "u1"
    assert args.func.__name__ == "_cmd_perception_delete_user"


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


def test_parse_run_discord_gateway():
    parser = _build_parser()
    args = parser.parse_args(["run", "discord-gateway", "--token", "abc", "--nats-url", "nats://x:4222"])
    assert args.command == "run"
    assert args.run_cmd == "discord-gateway"
    assert args.token == "abc"
    assert args.nats_url == "nats://x:4222"
    assert args.func.__name__ == "_cmd_run_discord_gateway"


def test_run_discord_gateway_requires_token():
    parser = _build_parser()
    args = parser.parse_args(["run", "discord-gateway", "--token", ""])
    with pytest.raises(SystemExit, match="DISCORD_BOT_TOKEN is required"):
        args.func(args)
