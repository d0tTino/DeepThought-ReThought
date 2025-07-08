import json

import pytest


def test_dashboard_creates_output(tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from tools import dashboard

    m1 = tmp_path / "m1.json"
    m2 = tmp_path / "m2.json"
    m1.write_text(json.dumps({"bleu": 0.5, "rouge_l": 0.2, "avg_latency": 1.0}))
    m2.write_text(json.dumps({"bleu": 0.6, "rouge_l": 0.3, "avg_latency": 1.2}))

    output = tmp_path / "out.png"

    dashboard.main([str(m1), str(m2), "--output", str(output)])

    assert output.exists()
