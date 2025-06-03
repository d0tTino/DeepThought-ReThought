#!/usr/bin/env python3
"""
Setup script for NATS JetStream streams needed for DeepThought reThought.
Run this script before running the tests to ensure all required streams are created.
"""

import asyncio
import logging
import sys
# import socket # Removed as check_nats_server_running is removed
from nats.aio.client import Client as NATS
from nats.js.api import StreamConfig
from nats.errors import TimeoutError

# Import configuration
import os
# Add src to sys.path to find the config module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.deepthought.config import get_nats_url, get_nats_stream_name, get_logging_level_str

# Configure logging
logging.basicConfig(
    level=get_logging_level_str().upper(), # Use level from config
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# def check_nats_server_running(host="localhost", port=4222): # REMOVED
#     """Check if NATS server is accessible at the given host and port.""" # REMOVED
#     pass # REMOVED

async def setup_jetstream():
    """Set up the JetStream streams needed for testing."""
    logger.info("Setting up JetStream streams for DeepThought reThought...")
    
    nats_url = get_nats_url()
    stream_name = get_nats_stream_name()

    # Connect to NATS
    nats_client = NATS()
    try:
        logger.info(f"Attempting to connect to NATS server at {nats_url}...")
        await nats_client.connect(servers=[nats_url])
        logger.info(f"Connected to NATS server at {nats_url}")
        
        # Create JetStream context
        js = nats_client.jetstream()
        
        # Define stream for DeepThought events
        logger.info(f"Configuring stream '{stream_name}' with subjects 'dtr.>'...")
        stream_config = StreamConfig(
            name=stream_name, # Use name from config
            subjects=["dtr.>"],  # All DeepThought subjects
            retention="limits",
            max_msgs_per_subject=10000, # Consider making this configurable too
            discard="old",
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
        
        logger.info(f"JetStream stream '{stream_name}' setup completed successfully.")
        
    except TimeoutError:
        logger.error(f"Timed out connecting to NATS server at {nats_url}.")
        logger.error("Please ensure your NATS server is running and JetStream is enabled (e.g., start with 'nats-server -js').")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to set up JetStream for stream '{stream_name}' at {nats_url}: {e}")
        if "Connection refused" in str(e):
            logger.error(f"Connection refused: NATS server is not running or not accessible at {nats_url}.")
        elif "Permissions Violation" in str(e) or "authorization violation" in str(e).lower():
            logger.error("NATS JetStream reported a permissions violation. This can sometimes happen if JetStream is not enabled on the server.")
            logger.error("Please ensure your NATS server is started with JetStream enabled (e.g., 'nats-server -js').")
        else:
            logger.error(f"An unexpected error occurred. Ensure NATS is running at {nats_url}, JetStream is enabled ('-js' flag).")
        sys.exit(1)
    finally:
        # Close the connection
        if nats_client.is_connected:
            await nats_client.drain()
            logger.info("Disconnected from NATS server")

if __name__ == "__main__":
    asyncio.run(setup_jetstream()) 