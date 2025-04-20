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
from nats.js import JetStreamContext
from nats.js.api import StreamConfig, ConsumerConfig, AckPolicy, DeliverPolicy, RetentionPolicy, StorageType, DiscardPolicy
from nats.js.errors import Error

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define subjects and stream name
SUBJECT_PREFIX = "deepthought.eda.tasks"
SUBJECT_REQUEST_NEW_TASK = f"{SUBJECT_PREFIX}.request.new"
SUBJECT_TASK_STATUS_UPDATE = f"{SUBJECT_PREFIX}.status.update"
SUBJECT_GET_FINAL_RESULT = f"{SUBJECT_PREFIX}.result.get"
STREAM_NAME = "EDA_TASKS_STREAM"

# Helper function to get NATS URL from environment variable
def get_nats_url() -> str:
    return os.getenv("NATS_URL", "nats://localhost:4222")

# The test function using an ephemeral consumer
@pytest.mark.asyncio
async def test_full_flow_direct_subscribe():
    """
    Test the full EDA flow using JetStream publish and a direct ephemeral subscribe.
    1. Publish a task request.
    2. Publish a status update.
    3. Publish the final result.
    4. Use js.subscribe to directly create an ephemeral consumer and receive messages.
    5. Verify all messages are received in order.
    """
    nc = None  # Define nc outside try
    
    try:
        # --- Connect NATS ---
        logger.info(f"Attempting to connect to NATS at {get_nats_url()}")
        nc = NATS()
        await nc.connect(servers=[get_nats_url()], connect_timeout=10)
        if not nc.is_connected:
            pytest.fail("NATS connection failed")
        logger.info("NATS connection successful.")
        
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
            nonlocal received_messages
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
            js = nc.jetstream(timeout=30.0, prefix="$JS.API")
            if not js:
                pytest.fail("Failed to get JetStream context inside test function.")
            logger.info("JetStream context obtained successfully inside test function.")
            
            # Ensure stream exists
            try:
                logger.info(f"Checking if stream '{STREAM_NAME}' exists...")
                await js.stream_info(STREAM_NAME)
                logger.info(f"Stream '{STREAM_NAME}' already exists.")
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
                await js.add_stream(stream_config)
                logger.info(f"Stream '{STREAM_NAME}' created successfully.")
            
            logger.info(f"Creating ephemeral push consumer by subscribing directly to '{SUBJECT_PREFIX}.>'...")
            ephemeral_sub = await js.subscribe(
                subject=f"{SUBJECT_PREFIX}.>",
                durable=None,  # Ephemeral
                cb=message_handler,
            )
            logger.info("Ephemeral consumer subscription successful.")
            subscription_ready.set() # Signal that subscription is ready
            
            # Wait briefly for subscription to be fully established
            await asyncio.sleep(0.5)
            
            # Publish messages
            logger.info(f"Publishing task request to '{SUBJECT_REQUEST_NEW_TASK}'...")
            await js.publish(SUBJECT_REQUEST_NEW_TASK, str(task_request_payload).encode())
            logger.info("Task request published.")
            
            logger.info(f"Publishing status update to '{SUBJECT_TASK_STATUS_UPDATE}'...")
            await js.publish(SUBJECT_TASK_STATUS_UPDATE, str(status_update_payload).encode())
            logger.info("Status update published.")
            
            logger.info(f"Publishing final result to '{SUBJECT_GET_FINAL_RESULT}'...")
            await js.publish(SUBJECT_GET_FINAL_RESULT, str(final_result_payload).encode())
            logger.info("Final result published.")
            
            # Wait for all messages to be received by the handler
            try:
                logger.info("Waiting for all messages to be received by ephemeral consumer...")
                await asyncio.wait_for(all_messages_received.wait(), timeout=10.0)
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
                SUBJECT_GET_FINAL_RESULT: str(final_result_payload)
            }
            received_payloads = {subj: data for subj, data in received_messages}
            
            assert received_payloads == expected_payloads, f"Received payloads do not match expected. Got: {received_payloads}"
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
                    await ephemeral_sub.unsubscribe()
                    logger.info("Ephemeral consumer unsubscribed.")
                except Exception as e:
                    logger.error(f"Error during ephemeral subscription cleanup: {e}", exc_info=True)
            logger.info("Test cleanup (subscription) finished.")
        
        logger.info("test_full_flow_direct_subscribe completed successfully.")
        # --- Existing Test Logic END ---
        
    finally:
        # --- Close NATS Connection ---
        if nc and nc.is_connected:
            logger.info("Closing NATS connection in test finally block.")
            await nc.close()  # Or await nc.drain()
            logger.info("NATS connection closed.")
        elif nc:
            logger.warning("NATS client existed but was not connected during teardown.")
        else:
            logger.warning("NATS client was not created during setup.")
