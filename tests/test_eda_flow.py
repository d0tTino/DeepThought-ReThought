# File: tests/test_eda_flow.py
"""
Tests for the EDA flow using NATS JetStream in DeepThought reThought.
(Simplified test with direct subscription and event sync)
"""
import asyncio
import logging
import os

import pytest
import pytest_asyncio
from nats.aio.client import Client as NATS
from nats.aio.errors import ErrTimeout
from nats.js.api import DeliverPolicy, DiscardPolicy, RetentionPolicy, StorageType, StreamConfig

from tests.helpers import nats_server_available

pytestmark = pytest.mark.nats

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Define subjects and stream name
# Use the generic 'dtr' prefix so that all tests share the same JetStream stream
# configuration.
SUBJECT_PREFIX = "dtr"
SUBJECT_REQUEST_NEW_TASK = f"{SUBJECT_PREFIX}.request.new"
SUBJECT_TASK_STATUS_UPDATE = f"{SUBJECT_PREFIX}.status.update"
SUBJECT_GET_FINAL_RESULT = f"{SUBJECT_PREFIX}.result.get"
STREAM_NAME = "deepthought_events"


# Helper function to get NATS URL from environment variable
def get_nats_url() -> str:
    return os.getenv("NATS_URL", "nats://localhost:4222")


# NATS connection fixture
@pytest_asyncio.fixture
async def nats_connection():
    """
    Fixture that creates a NATS client connection and tears it down after the test.
    This fixture only yields the NATS client, not the JetStream context.
    """
    if not nats_server_available(get_nats_url()):
        pytest.skip("NATS server not available")
    nc = None

    try:
        # Connect to NATS
        logger.info(f"Fixture: Connecting to NATS at {get_nats_url()}")
        nc = NATS()
        await nc.connect(servers=[get_nats_url()], connect_timeout=30)  # Increased timeout
        logger.info("Fixture: NATS connection successful")

        # Yield only the NATS client
        yield nc

    finally:
        # Close NATS connection
        if nc and nc.is_connected:
            logger.info("Fixture: Closing NATS connection")
            await nc.close()
            logger.info("Fixture: NATS connection closed")


# Simple test to check that the fixture works
@pytest.mark.asyncio
async def test_nats_connection_fixture(nats_connection):
    """Test that the NATS connection fixture is working properly."""
    assert nats_connection.is_connected, "NATS connection should be connected"
    logger.info("NATS connection fixture test passed")


# The test function using an ephemeral consumer
@pytest.mark.asyncio
async def test_full_flow_direct_subscribe(nats_connection, monkeypatch):
    """
    Test the full EDA flow using JetStream publish and a direct ephemeral subscribe.
    1. Publish a task request.
    2. Publish a status update.
    3. Publish the final result.
    4. Use js.subscribe to directly create an ephemeral consumer and receive messages.
    5. Verify all messages are received in order.
    """
    try:
        nc = nats_connection
        if not nc.is_connected:
            pytest.fail("NATS connection is not connected")
        logger.info("NATS connection from fixture is connected.")

        # Patch asyncio.sleep to speed up the test
        original_sleep = asyncio.sleep

        async def fast_sleep(*args, **kwargs):
            await original_sleep(0)

        monkeypatch.setattr(asyncio, "sleep", fast_sleep)

        # --- Existing Test Logic START ---
        logger.info("Starting test_full_flow_direct_subscribe...")

        # Define payloads
        task_request_payload = {"task_id": "task123", "data": "Sample data"}
        status_update_payload = {"task_id": "task123", "status": "Processing"}
        final_result_payload = {"task_id": "task123", "result": "Completed successfully"}

        received_messages = []
        subscription_ready = asyncio.Event()
        all_messages_received = asyncio.Event()

        async def message_handler(msg):
            subject = msg.subject
            data = msg.data.decode()
            logger.info(f"Ephemeral consumer received message on subject '{subject}': {data}")
            received_messages.append((subject, data))
            await msg.ack()
            logger.info(f"Acknowledged message on subject '{subject}'.")
            if len(received_messages) == 3:
                all_messages_received.set()

        ephemeral_sub = None
        try:
            # Get JetStream context inside test function
            logger.info("Getting JetStream context inside test function...")
            js = nc.jetstream(timeout=60.0)  # Increased timeout
            if not js:
                pytest.fail("Failed to get JetStream context inside test function.")
            logger.info("JetStream context obtained successfully inside test function.")

            # Ensure stream exists
            try:
                logger.info(f"Checking if stream '{STREAM_NAME}' exists...")
                await asyncio.wait_for(js.stream_info(STREAM_NAME), timeout=60.0)  # Added explicit timeout
                logger.info(f"Stream '{STREAM_NAME}' already exists.")
            except asyncio.TimeoutError:
                logger.error("Timeout while checking stream info")
                pytest.fail("Timeout while checking stream info")
            except Exception as e:
                logger.info(f"Stream '{STREAM_NAME}' does not exist, creating it... ({e})")
                stream_config = StreamConfig(
                    name=STREAM_NAME,
                    subjects=[f"{SUBJECT_PREFIX}.>"],
                    retention=RetentionPolicy.LIMITS,
                    storage=StorageType.MEMORY,
                    max_msgs_per_subject=100,
                    discard=DiscardPolicy.OLD,
                )
                try:
                    await asyncio.wait_for(js.add_stream(stream_config), timeout=60.0)  # Added explicit timeout
                    logger.info(f"Stream '{STREAM_NAME}' created successfully.")
                except asyncio.TimeoutError:
                    logger.error("Timeout while creating stream")
                    pytest.fail("Timeout while creating stream")
                except Exception as e:
                    logger.error(f"Failed to create stream: {e}")
                    pytest.fail(f"Failed to create stream: {e}")

            logger.info(f"Creating ephemeral push consumer by subscribing directly to '{SUBJECT_PREFIX}.>'...")
            ephemeral_sub = await js.subscribe(
                subject=f"{SUBJECT_PREFIX}.>",
                durable=None,  # Ephemeral
                cb=message_handler,
                stream=STREAM_NAME,  # Explicitly specify stream name
                deliver_policy=DeliverPolicy.NEW,  # Only receive new messages
            )
            logger.info("Ephemeral consumer subscription successful.")
            subscription_ready.set()  # Signal that subscription is ready

            # Wait briefly for subscription to be fully established
            await asyncio.sleep(1.0)  # Increased wait time

            # Publish messages
            logger.info(f"Publishing task request to '{SUBJECT_REQUEST_NEW_TASK}'...")
            await asyncio.wait_for(
                js.publish(SUBJECT_REQUEST_NEW_TASK, str(task_request_payload).encode()), timeout=30.0
            )  # Added explicit timeout
            logger.info("Task request published.")

            # Wait briefly between publishes
            await asyncio.sleep(0.5)

            logger.info(f"Publishing status update to '{SUBJECT_TASK_STATUS_UPDATE}'...")
            await asyncio.wait_for(
                js.publish(SUBJECT_TASK_STATUS_UPDATE, str(status_update_payload).encode()), timeout=30.0
            )  # Added explicit timeout
            logger.info("Status update published.")

            # Wait briefly between publishes
            await asyncio.sleep(0.5)

            logger.info(f"Publishing final result to '{SUBJECT_GET_FINAL_RESULT}'...")
            await asyncio.wait_for(
                js.publish(SUBJECT_GET_FINAL_RESULT, str(final_result_payload).encode()), timeout=30.0
            )  # Added explicit timeout
            logger.info("Final result published.")

            # Wait for all messages to be received by the handler
            try:
                logger.info("Waiting for all messages to be received by ephemeral consumer...")
                await asyncio.wait_for(all_messages_received.wait(), timeout=30.0)  # Increased timeout
                logger.info("All messages received.")
            except asyncio.TimeoutError:
                logger.error(f"Timeout waiting for messages. Received {len(received_messages)} messages.")
                pytest.fail(f"Timeout: Did not receive all 3 messages. Received: {received_messages}")

            # Verification
            assert len(received_messages) == 3, f"Expected 3 messages, got {len(received_messages)}"
            logger.info("Verifying received messages...")

            # We convert payloads to strings for comparison as they were published
            expected_payloads = {
                SUBJECT_REQUEST_NEW_TASK: str(task_request_payload),
                SUBJECT_TASK_STATUS_UPDATE: str(status_update_payload),
                SUBJECT_GET_FINAL_RESULT: str(final_result_payload),
            }
            received_payloads = {subj: data for subj, data in received_messages}

            assert (
                received_payloads == expected_payloads
            ), f"Received payloads do not match expected. Got: {received_payloads}"
            logger.info("Received messages verified successfully.")

        except ErrTimeout:
            logger.error("NATS operation timed out.")
            pytest.fail("NATS operation timed out.")
        except Exception as e:
            logger.error(f"An unexpected error occurred during the test: {e}", exc_info=True)
            pytest.fail(f"Test failed due to unexpected error: {e}")
        finally:
            logger.info("Cleaning up test resources (subscription)...")
            if ephemeral_sub:
                try:
                    logger.info("Unsubscribing ephemeral consumer...")
                    await asyncio.wait_for(ephemeral_sub.unsubscribe(), timeout=10.0)  # Added explicit timeout
                    logger.info("Ephemeral consumer unsubscribed.")
                except asyncio.TimeoutError:
                    logger.error("Timeout during unsubscribe")
                except Exception as e:
                    logger.error(f"Error during ephemeral subscription cleanup: {e}", exc_info=True)
            logger.info("Test cleanup (subscription) finished.")

        logger.info("test_full_flow_direct_subscribe completed successfully.")
        # --- Existing Test Logic END ---

    except Exception as e:
        logger.error(f"Unexpected error in top-level test: {e}", exc_info=True)
        pytest.fail(f"Test failed due to unexpected error: {e}")
