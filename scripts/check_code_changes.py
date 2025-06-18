#!/usr/bin/env python3

"""Determine whether a commit introduces code changes.

This helper is used by GitHub Actions to skip expensive CI runs when only
documentation or comment updates occur.
"""

import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Entry point for the code change checker."""

    # Determine the files changed compared to the base commit.
    BASE_REF = os.environ.get("GITHUB_BASE_REF")
    GITHUB_SHA = os.environ.get("GITHUB_SHA")
    BASE_SHA = None

    if BASE_REF and GITHUB_SHA:
        try:
            base_sha_result = subprocess.run(
                ["git", "merge-base", f"origin/{BASE_REF}", GITHUB_SHA],
                stdout=subprocess.PIPE,
                check=True,
            )
            BASE_SHA = base_sha_result.stdout.decode().strip()
        except subprocess.CalledProcessError:
            BASE_SHA = None

    if BASE_SHA:
        diff_args = ["git", "diff", "--name-only", BASE_SHA, GITHUB_SHA]
    else:
        diff_args = ["git", "diff", "--name-only", "HEAD^"]

    result = subprocess.run(diff_args, stdout=subprocess.PIPE, check=True)

    changed_files = [f for f in result.stdout.decode().splitlines() if f]

    # Common documentation file extensions to ignore
    doc_extensions = {".md", ".markdown", ".mdown", ".rst", ".txt"}

    # Helper to determine if a file is documentation
    def is_doc_file(path: str) -> bool:
        """Return True if the path looks like documentation."""
        p = Path(path)
        suffix = p.suffix.lower()
        return suffix in doc_extensions or path.startswith("docs/")

    # If all changed files are docs, tests are unnecessary
    if changed_files and all(is_doc_file(f) for f in changed_files):
        print("false")
        sys.exit(0)

    # Inspect the diff to see if code lines changed (ignoring comments and blank lines)
    if BASE_SHA:
        diff_code_args = ["git", "diff", "--unified=0", BASE_SHA, GITHUB_SHA]
    else:
        diff_code_args = ["git", "diff", "--unified=0", "HEAD^"]

    result = subprocess.run(diff_code_args, stdout=subprocess.PIPE, check=True)

    diff_lines = result.stdout.decode().splitlines()
    comment_prefixes = ("#", "//", "/*", "*", '"""', "'''")
    code_changed = False
    for line in diff_lines:
        if not line.startswith(("+", "-")):
            continue
        # Ignore diff metadata markers
        if line.startswith("+++") or line.startswith("---"):
            continue
        content = line[1:].strip()
        if not content:
            continue
        if any(content.startswith(prefix) for prefix in comment_prefixes):
            continue
        code_changed = True
        break

    print("true" if code_changed else "false")


if __name__ == "__main__":
    main()
