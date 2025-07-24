#!/usr/bin/env python3
"""Convert old trace logs to the interaction trace format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from deepthought.perception.social_perception import analyze as analyze_social


def _process(obj: dict | str, affinity: float) -> tuple[dict, float]:
    if isinstance(obj, dict):
        payload = obj.get("payload", obj)
    else:
        payload = obj
        obj = {"event": "CHAT_RAW", "payload": payload}

    text = None
    if isinstance(payload, dict) and "user_input" in payload:
        text = payload["user_input"]
    elif isinstance(payload, str):
        text = payload

    perception = None
    if text:
        perception = analyze_social(text)
        delta = perception.get("flirtation", 0.0) - (
            perception.get("avoidance", 0.0) + perception.get("manipulation", 0.0)
        )
        affinity += delta

    obj["perception"] = perception
    obj["affinity"] = affinity
    return obj, affinity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Path to the raw trace file")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output file")
    args = parser.parse_args()

    affinity = 0.0
    with args.input.open("r", encoding="utf-8") as fin, args.output.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            obj, affinity = _process(obj, affinity)
            json.dump(obj, fout)
            fout.write("\n")


if __name__ == "__main__":
    main()
