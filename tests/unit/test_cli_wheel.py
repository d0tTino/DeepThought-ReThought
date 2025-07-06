import os
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
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        subprocess.run(
            [str(pip), "install", "--no-deps", "--no-build-isolation", str(wheel)],
            check=True,
            env=env,
        )
        subprocess.run([str(dtrt), "init", "service", "demo"], cwd=tmp_path, check=True, env=env)
        assert (tmp_path / "src" / "deepthought" / "services" / "demo" / "service.py").exists()

        # verify the generated service can be imported and started
        script = """
import asyncio, sys
from pathlib import Path
import importlib.util
import types

fake_nats = types.ModuleType('nats')
fake_aio = types.ModuleType('aio')
fake_client = types.ModuleType('client')
setattr(fake_client, 'Client', object)
fake_msg_mod = types.ModuleType('msg')
setattr(fake_msg_mod, 'Msg', object)
fake_aio.client = fake_client
fake_aio.msg = fake_msg_mod
fake_js = types.ModuleType('js')
fake_js_client = types.ModuleType('client')
setattr(fake_js_client, 'JetStreamContext', object)
fake_js.client = fake_js_client
fake_nats.aio = fake_aio
fake_nats.js = fake_js
sys.modules.setdefault('nats', fake_nats)
sys.modules.setdefault('nats.aio', fake_aio)
sys.modules.setdefault('nats.aio.client', fake_client)
sys.modules.setdefault('nats.aio.msg', fake_msg_mod)
sys.modules.setdefault('nats.js', fake_js)
sys.modules.setdefault('nats.js.client', fake_js_client)

svc_path = Path(r'{svc}')
spec = importlib.util.spec_from_file_location('demo.service', svc_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
DemoService = module.DemoService

class DummyNATS:
    def __init__(self):
        self.is_connected = True

class DummyJS:
    async def publish(self, *a, **k):
        class Ack:
            seq = 1
            stream = 's'
        return Ack()
    async def subscribe(self, *a, **k):
        class Sub:
            async def unsubscribe(self):
                pass
        return Sub()

async def main():
    svc = DemoService(DummyNATS(), DummyJS())
    await svc.start()
    await svc.stop()

asyncio.run(main())
""".format(
            svc=str((tmp_path / "src" / "deepthought" / "services" / "demo" / "service.py").resolve())
        )
        run_py = venv_dir / "bin" / "python"
        script_file = tmp_path / "run_service.py"
        script_file.write_text(script)
        subprocess.run([str(run_py), str(script_file)], check=True, env=env)
    finally:
        shutil.rmtree(repo_root / "build", ignore_errors=True)
        shutil.rmtree(repo_root / "src" / "deepthought_rethought.egg-info", ignore_errors=True)
