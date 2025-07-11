import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from deepthought.metrics.prometheus import INPUTS_TOTAL
from deepthought.metrics_server import app


def test_metrics_endpoint_returns_data() -> None:
    INPUTS_TOTAL.labels(service="test").inc()
    client = TestClient(app)
    response = client.get("/metrics")
    assert response.status_code == 200
    assert b"inputs_total" in response.content
