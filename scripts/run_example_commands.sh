#!/usr/bin/env bash
set -euo pipefail

# Run all bash snippets from the example documentation.
# The docs file path is resolved relative to this script so it works when
# executed from any directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOC_FILE="${SCRIPT_DIR}/../docs/example_commands.md"

# Extract bash code blocks and execute them one by one.
awk '/^```bash/{flag=1;next}/^```/{flag=0}flag' "$DOC_FILE" | bash
