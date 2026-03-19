from __future__ import annotations

import ast
from pathlib import Path


TARGET_ROOTS = (
    Path("src/deepthought/services"),
    Path("src/deepthought/modules"),
)


def _iter_dtr_publish_violations() -> list[str]:
    violations: list[str] = []
    for root in TARGET_ROOTS:
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if not isinstance(func, ast.Attribute) or func.attr != "publish" or len(node.args) < 2:
                    continue
                subject = node.args[0]
                if not isinstance(subject, ast.Constant) or not isinstance(subject.value, str):
                    continue
                if not subject.value.startswith("dtr."):
                    continue
                payload_expr = ast.unparse(node.args[1])
                if "__dict__" not in payload_expr:
                    violations.append(f"{path}:{node.lineno} -> {subject.value} uses non-enveloped payload `{payload_expr}`")
    return violations


def test_dtr_subject_publishes_use_envelopes() -> None:
    violations = _iter_dtr_publish_violations()
    assert not violations, "Found dtr.* publishes without an envelope:\n" + "\n".join(violations)
