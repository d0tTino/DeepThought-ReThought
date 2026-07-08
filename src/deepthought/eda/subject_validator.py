"""Utilities for enforcing canonical EDA subjects in service wiring."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

from deepthought.eda.contracts import CanonicalSubjects, SubjectCanonicals, to_canonical_subject


@dataclass(frozen=True)
class SubjectViolation:
    location: str
    subject: str
    canonical_subject: str
    reason: str


CANONICAL_SUBJECTS: frozenset[str] = frozenset(
    value
    for name, value in vars(SubjectCanonicals).items()
    if name.isupper() and isinstance(value, str)
)


def is_canonical_subject(subject: str) -> bool:
    return subject in CANONICAL_SUBJECTS


def _resolve_event_subject(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    if value.startswith("EventSubjects."):
        name = value.split(".", 1)[1]
        resolved = getattr(CanonicalSubjects, name, None)
        return resolved if isinstance(resolved, str) else None
    return value


def validate_service_bindings(config: str | Path | dict[str, Any]) -> list[SubjectViolation]:
    if isinstance(config, (str, Path)):
        data = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    else:
        data = config
    violations: list[SubjectViolation] = []
    for service, binding in (data.get("service_bindings") or {}).items():
        for direction in ("subscribe", "publish"):
            for idx, entry in enumerate(binding.get(direction) or []):
                subject = _resolve_event_subject(entry.get("canonical_subject") or entry.get("event_subject"))
                if subject is None:
                    violations.append(SubjectViolation(f"{service}.{direction}[{idx}]", "", "", "missing subject"))
                    continue
                canonical = to_canonical_subject(subject)
                if subject != canonical or not is_canonical_subject(subject):
                    violations.append(SubjectViolation(f"{service}.{direction}[{idx}]", subject, canonical, "non-canonical binding subject"))
    return violations


class _ServiceSubjectVisitor(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.violations: list[SubjectViolation] = []

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.value, ast.Name) and node.value.id == "EventSubjects":
            subject = getattr(CanonicalSubjects, node.attr, None)
            if isinstance(subject, str):
                canonical = to_canonical_subject(subject)
                if subject != canonical or not is_canonical_subject(subject):
                    self.violations.append(SubjectViolation(f"{self.path}:{node.lineno}", subject, canonical, "non-canonical service subject"))
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str) and (node.value.startswith("dtr.") or node.value == "chat.raw"):
            canonical = to_canonical_subject(node.value)
            if node.value != canonical and canonical in CANONICAL_SUBJECTS:
                self.violations.append(SubjectViolation(f"{self.path}:{node.lineno}", node.value, canonical, "legacy literal service subject"))
        self.generic_visit(node)


def validate_service_modules(paths: Iterable[str | Path]) -> list[SubjectViolation]:
    violations: list[SubjectViolation] = []
    for raw_path in paths:
        path = Path(raw_path)
        files = [path] if path.is_file() else sorted(path.rglob("*.py"))
        for file_path in files:
            tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
            visitor = _ServiceSubjectVisitor(file_path)
            visitor.visit(tree)
            violations.extend(visitor.violations)
    return violations
