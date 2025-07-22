#!/usr/bin/env python3
"""Build and upload the dtrt-finetune package to PyPI."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def build(project_dir: Path) -> Path:
    """Build the wheel for the project located in ``project_dir``."""
    subprocess.run(["python", "-m", "build", "--wheel", str(project_dir)], check=True)
    dist_dir = project_dir / "dist"
    wheels = sorted(dist_dir.glob("*.whl"))
    if not wheels:
        raise RuntimeError(f"No wheel found in {dist_dir}")
    return wheels[-1]


def upload(wheel: Path, repository: str, skip_existing: bool) -> None:
    """Upload ``wheel`` to the given repository using twine."""
    cmd = ["twine", "upload", "--repository", repository]
    if skip_existing:
        cmd.append("--skip-existing")
    cmd.append(str(wheel))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish dtrt-finetune to PyPI")
    parser.add_argument(
        "--repository",
        default="pypi",
        help="Twine repository name (default: pypi)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Ignore existing files on the repository",
    )
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent.parent / "train"
    wheel = build(project_dir)
    upload(wheel, args.repository, args.skip_existing)


if __name__ == "__main__":  # pragma: no cover - manual script
    main()
