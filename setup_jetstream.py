#!/usr/bin/env python3
"""
Setup script for NATS JetStream streams needed for DeepThought reThought.
Run this script before running the tests to ensure all required streams are created.
"""

import asyncio
import logging
import os
import socket
import sys
from typing import Any, Dict
from urllib.parse import urlparse

from nats.aio.client import Client as NATS
from nats.errors import TimeoutError
from nats.js.api import (
    ConsumerConfig,
    DeliverPolicy,
    DiscardPolicy,
    RetentionPolicy,
    StorageType,
    StreamConfig,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# NATS server URL (can be overridden via environment variable)
NATS_URL = os.getenv("NATS_URL", "nats://localhost:4222")

# Durable consumers for the PERCEPTION stream
PERCEPTION_CONSUMERS = [
    "memory-perception-consumer",
    "analytics-perception-consumer",
    "perception-extract-listener",
]


_RETENTION_POLICY_ALIASES: Dict[str, RetentionPolicy] = {
    "limits": RetentionPolicy.LIMITS,
    "interest": RetentionPolicy.INTEREST,
    "workqueue": RetentionPolicy.WORK_QUEUE,
    "work_queue": RetentionPolicy.WORK_QUEUE,
    "work-queue": RetentionPolicy.WORK_QUEUE,
}


def _get_retention_policy(env_value: str | None) -> RetentionPolicy:
    """Return the :class:`RetentionPolicy` for ``env_value``.

    If ``env_value`` is ``None`` or invalid, ``RetentionPolicy.LIMITS`` is used.
    """

    if not env_value:
        return RetentionPolicy.LIMITS

    policy = _RETENTION_POLICY_ALIASES.get(env_value.strip().lower())
    if policy is None:
        logger.warning(
            "Unknown PERCEPTION_RETENTION_POLICY '%s'; defaulting to LIMITS",
            env_value,
        )
        return RetentionPolicy.LIMITS
    return policy


def _get_optional_int(env_var: str) -> int | None:
    """Return an ``int`` from ``env_var`` or ``None`` if unset/invalid."""

    raw = os.getenv(env_var)
    if raw is None or raw.strip() == "":
        return None
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid integer for %s: %s", env_var, raw)
        return None


def _build_perception_stream_config() -> StreamConfig:
    """Create the PERCEPTION stream configuration honoring environment overrides."""

    retention_policy = _get_retention_policy(os.getenv("PERCEPTION_RETENTION_POLICY"))
    max_msgs_per_subject = _get_optional_int("PERCEPTION_MAX_MSGS_PER_SUBJECT")
    max_msgs = _get_optional_int("PERCEPTION_MAX_MSGS")
    max_bytes = _get_optional_int("PERCEPTION_MAX_BYTES")
    max_age_seconds = _get_optional_int("PERCEPTION_MAX_AGE_SECONDS")

    config_kwargs: Dict[str, Any] = {
        "name": "PERCEPTION",
        "subjects": ["dtr.perception.>"],
        "retention": retention_policy,
        "storage": StorageType.FILE,
        "discard": DiscardPolicy.OLD,
    }

    if max_msgs_per_subject is not None:
        config_kwargs["max_msgs_per_subject"] = max_msgs_per_subject
    else:
        config_kwargs["max_msgs_per_subject"] = 10000

    if max_msgs is not None:
        config_kwargs["max_msgs"] = max_msgs
    if max_bytes is not None:
        config_kwargs["max_bytes"] = max_bytes
    if max_age_seconds is not None:
        config_kwargs["max_age"] = max_age_seconds * 1_000_000_000

    return StreamConfig(**config_kwargs)


def check_nats_server_running(url: str = NATS_URL) -> bool:
    """Check if NATS server is accessible at the given URL."""
    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 4222
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            result = sock.connect_ex((host, port))
            if result == 0:
                logger.info(f"NATS server appears to be running at {host}:{port}")
                return True
            else:
                logger.error(
                    f"No NATS server detected at {host}:{port} (connection refused)"
                )
                return False
    except Exception as e:
        logger.error(f"Error checking NATS server: {e}")
        return False


class JetStreamSetupError(Exception):
    """Raised when JetStream initialization fails."""


async def setup_jetstream() -> None:
    """Set up the JetStream streams needed for testing."""
    logger.info("Setting up JetStream streams for DeepThought reThought...")

    # First check if NATS server is running
    if not check_nats_server_running(NATS_URL):
        logger.error("NATS server does not appear to be running!")
        logger.error(
            "Please start a NATS server with JetStream enabled before running this script."
        )
        logger.error("Example command: 'nats-server -js'")
        raise JetStreamSetupError("NATS server unavailable")

    # Connect to NATS with retries
    nats_client = NATS()
    try:
        last_exc: Exception | None = None
        for attempt in range(1, 4):
            try:
                logger.info(
                    "Attempting to connect to NATS server (attempt %s/3)...",
                    attempt,
                )
                await nats_client.connect(servers=[NATS_URL])
                logger.info("Connected to NATS server")
                break
            except Exception as exc:  # pragma: no cover - network dependent
                last_exc = exc
                logger.warning("Connection attempt %s failed: %s", attempt, exc)
                if attempt < 3:
                    await asyncio.sleep(1)
        else:  # pragma: no cover - loop exhausted
            assert last_exc is not None
            raise last_exc

        # Create JetStream context
        js = nats_client.jetstream()

        # Define stream for DeepThought events
        stream_config = StreamConfig(
            name="deepthought_events",
            subjects=["dtr.>"],  # All DeepThought subjects
            retention=RetentionPolicy.LIMITS,
            storage=StorageType.MEMORY,
            max_msgs_per_subject=10000,
            discard=DiscardPolicy.OLD,
        )

        # Create or update the stream
        try:
            # Try to create the stream
            stream = await js.add_stream(config=stream_config)
            logger.info(f"Created JetStream stream: {stream.config.name}")
        except Exception as e:
            # If the stream already exists, update it
            logger.info(f"Stream might already exist, trying to update: {e}")
            stream = await js.update_stream(config=stream_config)
            logger.info(f"Updated JetStream stream: {stream.config.name}")

        # Define PERCEPTION stream with durability configuration overrides
        perception_stream = _build_perception_stream_config()

        try:
            await js.add_stream(config=perception_stream)
            logger.info("Created JetStream stream: PERCEPTION")
        except Exception as e:
            logger.info(f"PERCEPTION stream might already exist, trying to update: {e}")
            await js.update_stream(config=perception_stream)
            logger.info("Updated JetStream stream: PERCEPTION")

        # Create durable consumers for PERCEPTION stream
        for durable in PERCEPTION_CONSUMERS:
            consumer_config = ConsumerConfig(
                durable_name=durable,
                deliver_policy=DeliverPolicy.ALL,
            )
            try:
                await js.add_consumer("PERCEPTION", consumer_config)
                logger.info("Created durable consumer %s", durable)
            except Exception as e:  # pragma: no cover - consumer exists
                logger.info("Durable consumer %s might already exist: %s", durable, e)

        logger.info("JetStream setup completed successfully")

    except TimeoutError as e:
        logger.error(f"Timed out connecting to NATS server at {NATS_URL}.")
        logger.error(
            "Please ensure your NATS server is running and JetStream is enabled (e.g., start with 'nats-server -js')."
        )
        raise JetStreamSetupError("Timed out connecting to NATS") from e
    except Exception as e:
        logger.error(f"Failed to set up JetStream: {e}")
        if "Connection refused" in str(e):  # This check is good
            logger.error(
                f"Connection refused while trying to connect to NATS server at {NATS_URL}."
            )
            logger.error("Please ensure your NATS server is running.")
        elif (
            "Permissions Violation" in str(e)
            or "authorization violation" in str(e).lower()
        ):  # Added this
            logger.error(
                "NATS JetStream reported a permissions violation. This can sometimes happen if JetStream is not enabled on the server."
            )
            logger.error(
                "Please ensure your NATS server is started with JetStream enabled (e.g., 'nats-server -js')."
            )
        else:  # General advice for other errors
            logger.error(
                "An unexpected error occurred. Ensure NATS is running, JetStream is enabled ('-js' flag), and the server is accessible at %s."
                % NATS_URL
            )
        raise JetStreamSetupError("Failed to set up JetStream") from e
    finally:
        # Close the connection
        if nats_client.is_connected:
            await nats_client.drain()
            logger.info("Disconnected from NATS server")


def main() -> int:
    """CLI entry point for setup_jetstream."""
    try:
        asyncio.run(setup_jetstream())
        return 0
    except JetStreamSetupError as e:
        logger.error(e)
        return 1
    except Exception as e:  # pragma: no cover - unexpected errors
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
