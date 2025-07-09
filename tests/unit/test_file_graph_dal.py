import json
import networkx as nx
import sys
import types

sys.modules.setdefault("aiosqlite", types.ModuleType("aiosqlite"))
nats = types.ModuleType("nats")
nats.aio = types.ModuleType("aio")
nats.aio.client = types.ModuleType("client")
nats.aio.msg = types.ModuleType("msg")
nats.errors = types.ModuleType("errors")
nats.js = types.ModuleType("js")
nats.js.client = types.ModuleType("jsclient")
nats.aio.client.Client = object
nats.aio.msg.Msg = object
nats.errors.Error = Exception
nats.js.client.JetStreamContext = object
sys.modules.setdefault("nats", nats)
sys.modules.setdefault("nats.aio", nats.aio)
sys.modules.setdefault("nats.aio.client", nats.aio.client)
sys.modules.setdefault("nats.aio.msg", nats.aio.msg)
sys.modules.setdefault("nats.errors", nats.errors)
sys.modules.setdefault("nats.js", nats.js)
sys.modules.setdefault("nats.js.client", nats.js.client)
pydantic = types.ModuleType("pydantic")
pydantic.AnyUrl = str
pydantic.ValidationError = Exception
sys.modules.setdefault("pydantic", pydantic)
pydantic_settings = types.ModuleType("pydantic_settings")
pydantic_settings.BaseSettings = object
pydantic_settings.SettingsConfigDict = dict
sys.modules.setdefault("pydantic_settings", pydantic_settings)

import importlib.util
from pathlib import Path

package_root = Path(__file__).resolve().parents[2] / "src" / "deepthought"
deepthought_pkg = types.ModuleType("deepthought")
deepthought_pkg.__path__ = [str(package_root)]
services_pkg = types.ModuleType("deepthought.services")
services_pkg.__path__ = [str(package_root / "services")]
sys.modules.setdefault("deepthought", deepthought_pkg)
sys.modules.setdefault("deepthought.services", services_pkg)

spec = importlib.util.spec_from_file_location(
    "deepthought.services.file_graph_dal",
    package_root / "services" / "file_graph_dal.py",
)
file_graph_dal = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(file_graph_dal)
FileGraphDAL = file_graph_dal.FileGraphDAL


def test_add_interaction_creates_next_edge(tmp_path):
    graph_file = tmp_path / "g.json"
    dal = FileGraphDAL(str(graph_file))
    first = dal.add_interaction("hello")
    second = dal.add_interaction("world")
    assert dal._graph.has_edge(first, second)
    assert dal._graph[first][second]["relation"] == "next"


def test_get_recent_facts_returns_latest(tmp_path):
    graph_file = tmp_path / "g.json"
    dal = FileGraphDAL(str(graph_file))
    dal.add_interaction("a")
    dal.add_interaction("b")
    dal.add_interaction("c")
    assert dal.get_recent_facts(2) == ["b", "c"]


def test_corrupt_file_is_reset(tmp_path):
    graph_file = tmp_path / "g.json"
    # Write invalid JSON to simulate corruption
    graph_file.write_text("{invalid json}")
    dal = FileGraphDAL(str(graph_file))
    assert isinstance(dal._graph, nx.DiGraph)
    assert len(dal._graph.nodes) == 0
    # The file should now contain valid JSON for an empty graph
    with open(graph_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["nodes"] == []
