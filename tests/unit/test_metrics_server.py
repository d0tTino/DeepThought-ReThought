import builtins
import importlib
import sys

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from prometheus_client import CONTENT_TYPE_LATEST
from prometheus_client.parser import text_string_to_metric_families

from deepthought import metrics_server


def test_metrics_endpoint_text_format() -> None:
    client = TestClient(metrics_server.app)
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"] == CONTENT_TYPE_LATEST
    metrics = list(text_string_to_metric_families(response.text))
    assert metrics


def test_metrics_endpoint_without_prometheus(monkeypatch) -> None:
    with monkeypatch.context() as m:
        orig_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name.startswith("prometheus_client"):
                raise ModuleNotFoundError
            return orig_import(name, *args, **kwargs)

        m.setattr(builtins, "__import__", fake_import)
        m.delitem(sys.modules, "prometheus_client", raising=False)
        importlib.reload(metrics_server)
        client = TestClient(metrics_server.app)
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "text/plain; version=0.0.4; charset=utf-8"
        assert resp.content == b""

    importlib.reload(metrics_server)
