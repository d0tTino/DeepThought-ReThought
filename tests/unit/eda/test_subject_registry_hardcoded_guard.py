from __future__ import annotations

import ast
import re
from pathlib import Path

DTR_PATTERN = re.compile(r"\bdtr\.[a-z0-9_.>*-]+")

SCAN_PATHS = (
    Path("src/deepthought/eda"),
    Path("examples/orchestrator.yml"),
)

ALLOWED_PATHS = {
    Path("src/deepthought/eda/contracts/subject_registry.py"),
}


def _is_docstring_node(node: ast.Constant, parents: dict[ast.AST, ast.AST]) -> bool:
    parent = parents.get(node)
    if not isinstance(parent, ast.Expr):
        return False
    container = parents.get(parent)
    if isinstance(container, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
        return bool(container.body) and container.body[0] is parent
    return False


def _scan_python(path: Path) -> list[str]:
    violations: list[str] = []
    tree = ast.parse(path.read_text(), filename=str(path))
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if _is_docstring_node(node, parents):
                continue
            if DTR_PATTERN.search(node.value):
                violations.append(f"{path}:{node.lineno} -> {node.value}")
    return violations


def _scan_text(path: Path) -> list[str]:
    violations: list[str] = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        match = DTR_PATTERN.search(line)
        if match:
            violations.append(f"{path}:{lineno} -> {match.group(0)}")
    return violations


def test_no_hardcoded_dtr_subjects_outside_registry_or_migration_map() -> None:
    violations: list[str] = []

    for scan_path in SCAN_PATHS:
        if scan_path.is_file():
            paths = [scan_path]
        else:
            paths = sorted(scan_path.rglob("*.py"))

        for path in paths:
            if path in ALLOWED_PATHS:
                continue
            if path.suffix == ".py":
                violations.extend(_scan_python(path))
            else:
                violations.extend(_scan_text(path))

    assert not violations, "Hardcoded dtr.* subject literals detected:\n" + "\n".join(violations)
