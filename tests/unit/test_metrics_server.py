from fastapi.testclient import TestClient
from prometheus_client import CONTENT_TYPE_LATEST
from prometheus_client.parser import text_string_to_metric_families

from deepthought.metrics_server import app


def test_metrics_endpoint_text_format() -> None:
    client = TestClient(app)
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"] == CONTENT_TYPE_LATEST
    metrics = list(text_string_to_metric_families(response.text))
    assert metrics
