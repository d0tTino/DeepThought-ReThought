#!/usr/bin/env python3
"""Verify encoder model versions against a whitelist."""
from __future__ import annotations

import ast
import json
from pathlib import Path

# Location of PerceptionConfig
CONFIG_PATH = Path("src/deepthought/services/perception/config.py")
WHITELIST_PATH = Path(__file__).with_name("model_version_whitelist.json")


def load_config_defaults() -> dict[str, str]:
    tree = ast.parse(CONFIG_PATH.read_text())
    defaults: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
            if name in {"text_model", "audio_model", "video_model"}:
                defaults[name] = ast.literal_eval(node.value)
    return defaults


def main() -> None:
    whitelist = json.loads(WHITELIST_PATH.read_text())
    current = load_config_defaults()

    mismatches = [f"{k}: expected {whitelist[k]}, found {v}" for k, v in current.items() if whitelist.get(k) != v]
    if mismatches:
        raise SystemExit("Model version check failed:\n" + "\n".join(mismatches))
    print("Model versions verified against whitelist.")


if __name__ == "__main__":
    main()
