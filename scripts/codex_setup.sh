#!/usr/bin/env bash
set -euo pipefail

# Install dependencies
pip install -r requirements-ci.txt
pip install -e .

CONTAINER_NAME="dtr_nats"

# Start NATS if container is not already running
if ! docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    scripts/start_nats.sh
fi

# Ensure JetStream stream exists
python setup_jetstream.py

# Determine if code changes require running tests and linters
RUN_CHECKS=$(python scripts/check_code_changes.py)
if [[ "$RUN_CHECKS" == "true" ]]; then
    pre-commit run --all-files
    flake8 src tests
    PYTHONPATH=src pytest
else
    echo "No code changes detected; skipping linters and tests."
fi

# Tear down NATS container
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    docker rm -f ${CONTAINER_NAME}
fi
