import sys
from types import ModuleType

# Provide fake pyperplan and l2p modules to avoid optional dependencies
fake_py = ModuleType("pyperplan")
fake_parser_mod = ModuleType("parser")
setattr(fake_parser_mod, "Parser", object)
fake_py.pddl = ModuleType("pddl")
fake_py.pddl.parser = fake_parser_mod
fake_planner_mod = ModuleType("planner")
setattr(fake_planner_mod, "_ground", lambda p: p)
fake_py.planner = fake_planner_mod
fake_search_mod = ModuleType("search")
setattr(fake_search_mod, "breadth_first_search", lambda task: [])
fake_py.search = fake_search_mod
sys.modules.setdefault("pyperplan", fake_py)
sys.modules.setdefault("pyperplan.pddl", fake_py.pddl)
sys.modules.setdefault("pyperplan.pddl.parser", fake_parser_mod)
sys.modules.setdefault("pyperplan.planner", fake_planner_mod)
sys.modules.setdefault("pyperplan.search", fake_search_mod)

fake_l2p = ModuleType("l2p")
fake_l2p_utils = ModuleType("utils")
setattr(fake_l2p_utils, "parse_domain", lambda p: None)
setattr(fake_l2p_utils, "parse_problem", lambda p: None)
fake_l2p.utils = fake_l2p_utils
sys.modules.setdefault("l2p", fake_l2p)
sys.modules.setdefault("l2p.utils", fake_l2p_utils)


import asyncio
import json
from types import SimpleNamespace

import pytest

from tests.unit.test_orchestrator import DummyService, _load_orchestrator_module


@pytest.mark.asyncio
async def test_load_config_json_and_yaml(tmp_path):
    orch = _load_orchestrator_module()
    cfg_json = tmp_path / "cfg.json"
    cfg_json.write_text('{"services": ["a", "b"]}', encoding="utf-8")
    cfg = orch._load_config(str(cfg_json))
    assert cfg["services"] == ["a", "b"]

    cfg_yaml = tmp_path / "cfg.yaml"
    cfg_yaml.write_text("services:\n  - foo\n", encoding="utf-8")
    cfg2 = orch._load_config(str(cfg_yaml))
    assert cfg2["services"] == ["foo"]


@pytest.mark.asyncio
async def test_run_starts_and_stops_service(monkeypatch, tmp_path):
    orch = _load_orchestrator_module()
    created = {}

    orig_init = DummyService.__init__

    def capture_init(self, nc, js):
        orig_init(self, nc, js)
        created["instance"] = self

    monkeypatch.setattr(DummyService, "__init__", capture_init)

    ep = SimpleNamespace(name="dummy", load=lambda: DummyService)
    monkeypatch.setattr(
        orch.metadata,
        "entry_points",
        lambda: SimpleNamespace(select=lambda **k: [ep]),
    )

    class DummyNATS:
        def __init__(self) -> None:
            self.is_connected = True

        async def drain(self):
            pass

    class DummyJS:
        pass

    async def fake_connect():
        return DummyNATS(), DummyJS()

    monkeypatch.setattr(orch, "_connect_nats", fake_connect)

    monkeypatch.setattr(asyncio.Event, "wait", lambda self: asyncio.sleep(0))

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("services:\n  - dummy\n", encoding="utf-8")

    await orch.run(str(cfg))

    service = created.get("instance")
    assert service is not None
    assert service.started is True
    assert service.stopped is True
