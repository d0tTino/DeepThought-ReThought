#!/usr/bin/env bash
# Start a local NATS server with JetStream using Docker.
# The server will be exposed on ports 4222 (client) and 8222 (monitoring).
# A container named 'dtr_nats' will be created in detached mode.

set -e

CONTAINER_NAME="dtr_nats"
IMAGE="nats:latest"

# Run the container in detached mode
# --rm ensures it is cleaned up when stopped
# -js enables JetStream

exec docker run --rm -d --name "$CONTAINER_NAME" -p 4222:4222 -p 8222:8222 "$IMAGE" -js

