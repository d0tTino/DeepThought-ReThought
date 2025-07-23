from deepthought.risk import score_commit_risk
from deepthought.neuralsymbolic import security_hotfix


def test_security_hotfix_invocation(monkeypatch):
    calls = []

    def fake_model(msg: str) -> float:
        calls.append(msg)
        return 0.42

    monkeypatch.setattr(security_hotfix, "model_func", fake_model)
    monkeypatch.setattr(security_hotfix, "predicate", security_hotfix._build())

    score = score_commit_risk("Fix CVE-9999")
    assert score == 0.42
    assert calls == ["Fix CVE-9999"]


def test_score_commit_risk_default():
    assert score_commit_risk("initial commit") == 0.0
