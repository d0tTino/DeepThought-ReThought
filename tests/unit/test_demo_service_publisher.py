import importlib.util
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

# Stub out heavy dependencies
fake_nats = types.ModuleType("nats")
import importlib.machinery

fake_nats.__spec__ = importlib.machinery.ModuleSpec("nats", loader=None)
fake_nats.errors = types.SimpleNamespace(TimeoutError=Exception)

aio_mod = types.ModuleType("nats.aio")
client_mod = types.ModuleType("nats.aio.client")
client_mod.Client = object
aio_mod.client = client_mod

js_client_mod = types.ModuleType("nats.js.client")
js_client_mod.JetStreamContext = object

msg_mod = types.ModuleType("nats.aio.msg")
msg_mod.Msg = object

sys.modules.setdefault("nats", fake_nats)
sys.modules.setdefault("nats.aio", aio_mod)
sys.modules.setdefault("nats.aio.client", client_mod)
sys.modules.setdefault("nats.aio.msg", msg_mod)
sys.modules.setdefault("nats.js.client", js_client_mod)

dummy_db_manager = types.ModuleType("db_manager")
dummy_db_manager.DBManager = object
sys.modules.setdefault("deepthought.services.db_manager", dummy_db_manager)
modules = {
    "deepthought.services.file_graph_dal": "FileGraphDAL",
    "deepthought.services.hierarchical_service": "HierarchicalService",
    "deepthought.services.cognitive_core_service": "CognitiveCoreService",
    "deepthought.services.persona_manager": "PersonaManager",
    "deepthought.services.social_graph_service": "SocialGraphService",
}
for name, class_name in modules.items():
    mod = types.ModuleType(name.split(".")[-1])
    setattr(mod, class_name, object)
    sys.modules.setdefault(name, mod)


class StubNATS:
    def __init__(self):
        self.is_connected = True


class StubJS:
    def __init__(self):
        self.published = []

    async def publish(self, subject, data, timeout=10.0):
        self.published.append((subject, data, timeout))

        class Ack:
            def __init__(self):
                self.seq = 1
                self.stream = "demo"

        return Ack()


@pytest.mark.asyncio
async def test_demo_service_publisher_uses_jetstream(tmp_path):
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
    service_dir = tmp_path / "src" / "deepthought" / "services" / "demo"
    publisher_path = service_dir / "publisher.py"
    spec = importlib.util.spec_from_file_location("demo.publisher", publisher_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    DemoServicePublisher = module.DemoServicePublisher

    nc = StubNATS()
    js = StubJS()
    pub = DemoServicePublisher(nc, js)

    await pub.publish_example("demo.test", "payload")

    assert js.published == [("demo.test", b"payload", 10.0)]
