#!/usr/bin/env python3
"""Verify third-party model license entries."""
from __future__ import annotations

import ast
import json
from pathlib import Path

LICENSE_FILE = Path(__file__).resolve().parents[1] / "docs" / "licenses.md"
CONFIG_PATH = Path("src/deepthought/services/perception/config.py")
WHITELIST_PATH = Path(__file__).with_name("model_version_whitelist.json")


def parse_table(text: str) -> dict[str, str]:
    """Parse Markdown table rows into a mapping of component to license."""
    entries: dict[str, str] = {}
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        parts = [p.strip() for p in line.strip().split("|")[1:-1]]
        if len(parts) < 2:
            continue
        component, license_name = parts[0], parts[1]
        entries[component] = license_name
    return entries


def load_config_models() -> dict[str, str]:
    """Return default model names from :class:`PerceptionConfig`."""
    tree = ast.parse(CONFIG_PATH.read_text(encoding="utf-8"))
    models: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
            if name in {"text_model", "audio_model", "video_model"}:
                models[name] = ast.literal_eval(node.value)
    return models


def main() -> int:
    """Entry point for the license verifier."""
    if not LICENSE_FILE.exists():
        print(f"License file not found: {LICENSE_FILE}")
        return 1

    content = LICENSE_FILE.read_text(encoding="utf-8")
    entries = parse_table(content)

    models = load_config_models()
    whitelist = json.loads(WHITELIST_PATH.read_text(encoding="utf-8"))
    license_info = whitelist.get("licenses", {})

    missing: list[str] = []
    for field in models:
        info = license_info.get(field)
        if info is None:
            missing.append(f"{field} missing from license whitelist")
            continue
        component = info["component"]
        expected = info["license"]
        if entries.get(component) != expected:
            missing.append(f"{component} ({expected})")

    if missing:
        print("Missing or incorrect license entries:")
        for item in missing:
            print(f" - {item}")
        return 1

    print("All required license entries present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
