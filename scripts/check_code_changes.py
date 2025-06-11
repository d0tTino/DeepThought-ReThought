#!/usr/bin/env python3

"""Determine whether a commit introduces code changes.

This helper is used by GitHub Actions to skip expensive CI runs when only
documentation or comment updates occur.
"""

import subprocess
import sys
from pathlib import Path

# Determine the files changed compared to origin/main. Fall back to HEAD^ if
# the remote is unavailable (e.g. local testing without origin).
try:
    result = subprocess.run(
        ["git", "diff", "--name-only", "origin/main...HEAD"],
        stdout=subprocess.PIPE,
        check=True,
    )
except subprocess.CalledProcessError:
    result = subprocess.run(
        ["git", "diff", "--name-only", "HEAD^"],
        stdout=subprocess.PIPE,
        check=True,
    )

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
try:
    result = subprocess.run(
        ["git", "diff", "--unified=0", "origin/main...HEAD"],
        stdout=subprocess.PIPE,
        check=True,
    )
except subprocess.CalledProcessError:
    result = subprocess.run(
        ["git", "diff", "--unified=0", "HEAD^"],
        stdout=subprocess.PIPE,
        check=True,
    )

diff_lines = result.stdout.decode().splitlines()
comment_prefixes = ("#", "//", "/*", "*", '"""', "'''")
code_changed = False
for line in diff_lines:
    if not line.startswith(('+', '-')):
        continue
    # Ignore diff metadata markers
    if line.startswith('+++') or line.startswith('---'):
        continue
    content = line[1:].strip()
    if not content:
        continue
    if any(content.startswith(prefix) for prefix in comment_prefixes):
        continue
    code_changed = True
    break

print("true" if code_changed else "false")
