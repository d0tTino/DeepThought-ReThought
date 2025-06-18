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
from urllib.parse import urlparse

from nats.aio.client import Client as NATS
from nats.errors import TimeoutError
from nats.js.api import DiscardPolicy, RetentionPolicy, StorageType, StreamConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# NATS server URL (can be overridden via environment variable)
NATS_URL = os.getenv("NATS_URL", "nats://localhost:4222")


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
                logger.error(f"No NATS server detected at {host}:{port} (connection refused)")
                return False
    except Exception as e:
        logger.error(f"Error checking NATS server: {e}")
        return False


async def setup_jetstream():
    """Set up the JetStream streams needed for testing."""
    logger.info("Setting up JetStream streams for DeepThought reThought...")

    # First check if NATS server is running
    if not check_nats_server_running(NATS_URL):
        logger.error("NATS server does not appear to be running!")
        logger.error("Please start a NATS server with JetStream enabled before running this script.")
        logger.error("Example command: 'nats-server -js'")
        sys.exit(1)

    # Connect to NATS
    nats_client = NATS()
    try:
        logger.info("Attempting to connect to NATS server...")
        await nats_client.connect(servers=[NATS_URL])
        logger.info("Connected to NATS server")

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

        logger.info("JetStream setup completed successfully")

    except TimeoutError:
        logger.error(f"Timed out connecting to NATS server at {NATS_URL}.")
        logger.error(
            "Please ensure your NATS server is running and JetStream is enabled (e.g., start with 'nats-server -js')."
        )
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to set up JetStream: {e}")
        if "Connection refused" in str(e):  # This check is good
            logger.error(f"Connection refused while trying to connect to NATS server at {NATS_URL}.")
            logger.error("Please ensure your NATS server is running.")
        elif "Permissions Violation" in str(e) or "authorization violation" in str(e).lower():  # Added this
            logger.error(
                "NATS JetStream reported a permissions violation. This can sometimes happen if JetStream is not enabled on the server."
            )
            logger.error("Please ensure your NATS server is started with JetStream enabled (e.g., 'nats-server -js').")
        else:  # General advice for other errors
            logger.error(
                "An unexpected error occurred. Ensure NATS is running, JetStream is enabled ('-js' flag), and the server is accessible at %s."
                % NATS_URL
            )
        sys.exit(1)
    finally:
        # Close the connection
        if nats_client.is_connected:
            await nats_client.drain()
            logger.info("Disconnected from NATS server")


if __name__ == "__main__":
    asyncio.run(setup_jetstream())
