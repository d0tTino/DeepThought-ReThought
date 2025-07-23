from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from deepthought.psl import RiskScorer


def test_risk_scorer(tmp_path: Path) -> None:
    cfg = {
        "model": {"weights": {"lines_added": 0.1, "lines_deleted": 0.2}},
        "threshold": 1.0,
    }
    cfg_path = tmp_path / "model.yml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    scorer = RiskScorer.from_file(cfg_path)

    low = {"lines_added": 5, "lines_deleted": 2}
    assert scorer.score(low) == pytest.approx(0.9)
    assert not scorer.is_high_risk(low)

    high = {"lines_added": 8, "lines_deleted": 3}
    assert scorer.is_high_risk(high)
