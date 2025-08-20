#!/usr/bin/env python3
"""Verify required third-party license entries."""
from __future__ import annotations

from pathlib import Path

LICENSE_FILE = Path(__file__).resolve().parents[1] / "docs" / "licenses.md"
REQUIRED_ENTRIES = {
    "WavLM": "MIT",
    "WavLM checkpoints": "MIT",
    "CLAP": "CC0",
    "CLAP checkpoints": "CC0",
    "SigLIP": "Apache-2.0",
    "SigLIP checkpoints": "Apache-2.0",
}


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


def main() -> int:
    """Entry point for the license verifier."""
    if not LICENSE_FILE.exists():
        print(f"License file not found: {LICENSE_FILE}")
        return 1

    content = LICENSE_FILE.read_text(encoding="utf-8")
    entries = parse_table(content)

    missing = [
        f"{component} ({expected})"
        for component, expected in REQUIRED_ENTRIES.items()
        if entries.get(component) != expected
    ]

    if missing:
        print("Missing or incorrect license entries:")
        for item in missing:
            print(f" - {item}")
        return 1

    print("All required license entries present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
