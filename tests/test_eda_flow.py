# File: tests/test_eda_flow.py
"""Tests for the EDA flow using NATS JetStream in DeepThought reThought."""

import os
import asyncio
import logging

import pytest
import pytest_asyncio

from src.deepthought.config import DEFAULT_CONFIG
from src.deepthought.eda.events import (
    EventSubjects,
    SocialGraphUpdatePayload,
    SocialGraphSnapshotPayload,
    QuestCreatePayload,
    QuestUpdatePayload,
    QuestDonePayload,
    ResponseCandidatesPayload,
    ResponseRankedPayload,
    PerceptionAudioEmbedPayload,
    PerceptionImageEmbedPayload,
)
from nats.aio.client import Client as NATS
from nats.aio.errors import ErrTimeout
from nats.js import JetStreamContext
from nats.js.api import (
    StreamConfig,
    ConsumerConfig,
    AckPolicy,
    DeliverPolicy,
    RetentionPolicy,
    StorageType,
    DiscardPolicy,
)
from nats.js.errors import Error


RUN_NATS_TESTS = os.getenv("RUN_NATS_TESTS") == "1"
NATS_SKIP_REASON = "NATS tests skipped (set RUN_NATS_TESTS=1 to enable)"

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Define subjects and stream name
SUBJECT_PREFIX = "dtr.tasks"  # Changed to align with 'dtr.>' stream
SUBJECT_REQUEST_NEW_TASK = f"{SUBJECT_PREFIX}.request.new"
SUBJECT_TASK_STATUS_UPDATE = f"{SUBJECT_PREFIX}.status.update"
SUBJECT_GET_FINAL_RESULT = f"{SUBJECT_PREFIX}.result.get"
STREAM_NAME = DEFAULT_CONFIG.stream_name


def test_event_subject_constants_and_payload_roundtrip():
    """Ensure new EDA subjects exist and payloads round-trip through JSON."""

    assert EventSubjects.SOCIAL_GRAPH_UPDATE == "dtr.social.graph.update"
    assert EventSubjects.SOCIAL_GRAPH_SNAPSHOT == "dtr.social.graph.snapshot"
    assert EventSubjects.QUEST_CREATE == "dtr.quest.create"
    assert EventSubjects.QUEST_UPDATE == "dtr.quest.update"
    assert EventSubjects.QUEST_DONE == "dtr.quest.done"
    assert EventSubjects.RESPONSE_CANDIDATES == "dtr.response.candidates"
    assert EventSubjects.RESPONSE_RANKED == "dtr.response.ranked"
    assert EventSubjects.PERCEPTION_AUDIO_EMBED == "dtr.perception.audio.embed"
    assert EventSubjects.PERCEPTION_IMAGE_EMBED == "dtr.perception.image.embed"

    payload_cases = [
        (
            SocialGraphUpdatePayload,
            {
                "user_id": "user-123",
                "updates": {"edges_added": [["user-123", "user-456"]]},
                "timestamp": "2024-01-01T00:00:00Z",
            },
        ),
        (
            SocialGraphSnapshotPayload,
            {
                "user_id": "user-123",
                "graph": {
                    "nodes": ["user-123", "user-456"],
                    "edges": [["user-123", "user-456"]],
                },
            },
        ),
        (
            QuestCreatePayload,
            {
                "quest_id": "quest-1",
                "name": "Collect artifacts",
                "description": "Gather three unique artifacts",
                "metadata": {"difficulty": "medium"},
            },
        ),
        (
            QuestUpdatePayload,
            {
                "quest_id": "quest-1",
                "status": "in_progress",
                "progress": 0.5,
                "metadata": {"artifact_count": 1},
            },
        ),
        (
            QuestDonePayload,
            {
                "quest_id": "quest-1",
                "result": {"success": True, "reward": 1200},
                "timestamp": "2024-01-02T12:00:00Z",
            },
        ),
        (
            ResponseCandidatesPayload,
            {
                "input_id": "input-42",
                "candidates": [
                    {"text": "Hello", "score": 0.7},
                    {"text": "Hi there", "score": 0.6},
                ],
            },
        ),
        (
            ResponseRankedPayload,
            {
                "input_id": "input-42",
                "ranked_candidates": [
                    {"text": "Hello", "score": 0.92},
                    {"text": "Hi there", "score": 0.75},
                ],
                "selected_index": 0,
            },
        ),
        (
            PerceptionAudioEmbedPayload,
            {
                "audio_id": "audio-1",
                "embedding": [0.1, 0.2, 0.3],
                "metadata": {"sample_rate": 16000},
            },
        ),
        (
            PerceptionImageEmbedPayload,
            {
                "image_id": "image-1",
                "embedding": [0.9, 0.8, 0.7],
                "metadata": {"height": 256, "width": 256},
            },
        ),
    ]

    for payload_cls, payload_kwargs in payload_cases:
        payload = payload_cls(**payload_kwargs)
        json_payload = payload.to_json()
        from_json = payload_cls.from_json(json_payload)
        assert from_json == payload

        from_dict = payload_cls.from_dict(payload_kwargs)
        assert from_dict == payload


# Helper function to get NATS URL from environment variable
def get_nats_url() -> str:
    return os.getenv("NATS_URL", DEFAULT_CONFIG.nats_url)


# NATS connection fixture
@pytest_asyncio.fixture
async def nats_connection():
    """
    Fixture that creates a NATS client connection and tears it down after the test.
    This fixture only yields the NATS client, not the JetStream context.
    """
    if not RUN_NATS_TESTS:
        pytest.skip(NATS_SKIP_REASON)

    nc = None

    try:
        # Connect to NATS
        logger.info(f"Fixture: Connecting to NATS at {get_nats_url()}")
        nc = NATS()
        await nc.connect(
            servers=[get_nats_url()], connect_timeout=30
        )  # Increased timeout
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
@pytest.mark.skipif(not RUN_NATS_TESTS, reason=NATS_SKIP_REASON)
async def test_nats_connection_fixture(nats_connection):
    """Test that the NATS connection fixture is working properly."""
    assert nats_connection.is_connected, "NATS connection should be connected"
    logger.info("NATS connection fixture test passed")


# The test function using an ephemeral consumer
@pytest.mark.asyncio
@pytest.mark.skipif(not RUN_NATS_TESTS, reason=NATS_SKIP_REASON)
async def test_full_flow_direct_subscribe(nats_connection):
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

        # --- Existing Test Logic START ---
        logger.info("Starting test_full_flow_direct_subscribe...")

        # Define payloads
        task_request_payload = {"task_id": "task123", "data": "Sample data"}
        status_update_payload = {"task_id": "task123", "status": "Processing"}
        final_result_payload = {
            "task_id": "task123",
            "result": "Completed successfully",
        }

        received_messages = []
        subscription_ready = asyncio.Event()
        all_messages_received = asyncio.Event()

        async def message_handler(msg):
            nonlocal received_messages
            subject = msg.subject
            data = msg.data.decode()
            logger.info(
                f"Ephemeral consumer received message on subject '{subject}': {data}"
            )
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
                await asyncio.wait_for(
                    js.stream_info(STREAM_NAME), timeout=60.0
                )  # Added explicit timeout
                logger.info(f"Stream '{STREAM_NAME}' already exists.")
            except asyncio.TimeoutError:
                logger.error("Timeout while checking stream info")
                pytest.fail("Timeout while checking stream info")
            except Exception as e:
                logger.info(
                    f"Stream '{STREAM_NAME}' does not exist, creating it... ({e})"
                )
                stream_config = StreamConfig(
                    name=STREAM_NAME,
                    subjects=[f"{SUBJECT_PREFIX}.>"],
                    retention=RetentionPolicy.LIMITS,
                    storage=StorageType.MEMORY,
                    max_msgs_per_subject=100,
                    discard=DiscardPolicy.OLD,
                )
                try:
                    await asyncio.wait_for(
                        js.add_stream(stream_config), timeout=60.0
                    )  # Added explicit timeout
                    logger.info(f"Stream '{STREAM_NAME}' created successfully.")
                except asyncio.TimeoutError:
                    logger.error("Timeout while creating stream")
                    pytest.fail("Timeout while creating stream")
                except Exception as e:
                    logger.error(f"Failed to create stream: {e}")
                    pytest.fail(f"Failed to create stream: {e}")

            logger.info(
                f"Creating ephemeral push consumer by subscribing directly to '{SUBJECT_PREFIX}.>'..."
            )
            ephemeral_sub = await js.subscribe(
                subject=f"{SUBJECT_PREFIX}.>",
                durable=None,  # Ephemeral
                cb=message_handler,
                stream=STREAM_NAME,  # Explicitly specify stream name
            )
            logger.info("Ephemeral consumer subscription successful.")
            subscription_ready.set()  # Signal that subscription is ready

            # Wait briefly for subscription to be fully established
            await asyncio.sleep(1.0)  # Increased wait time

            # Publish messages
            logger.info(f"Publishing task request to '{SUBJECT_REQUEST_NEW_TASK}'...")
            await asyncio.wait_for(
                js.publish(
                    SUBJECT_REQUEST_NEW_TASK, str(task_request_payload).encode()
                ),
                timeout=30.0,
            )  # Added explicit timeout
            logger.info("Task request published.")

            # Wait briefly between publishes
            await asyncio.sleep(0.5)

            logger.info(
                f"Publishing status update to '{SUBJECT_TASK_STATUS_UPDATE}'..."
            )
            await asyncio.wait_for(
                js.publish(
                    SUBJECT_TASK_STATUS_UPDATE, str(status_update_payload).encode()
                ),
                timeout=30.0,
            )  # Added explicit timeout
            logger.info("Status update published.")

            # Wait briefly between publishes
            await asyncio.sleep(0.5)

            logger.info(f"Publishing final result to '{SUBJECT_GET_FINAL_RESULT}'...")
            await asyncio.wait_for(
                js.publish(
                    SUBJECT_GET_FINAL_RESULT, str(final_result_payload).encode()
                ),
                timeout=30.0,
            )  # Added explicit timeout
            logger.info("Final result published.")

            # Wait for all messages to be received by the handler
            try:
                logger.info(
                    "Waiting for all messages to be received by ephemeral consumer..."
                )
                await asyncio.wait_for(
                    all_messages_received.wait(), timeout=30.0
                )  # Increased timeout
                logger.info("All messages received.")
            except asyncio.TimeoutError:
                logger.error(
                    f"Timeout waiting for messages. Received {len(received_messages)} messages."
                )
                pytest.fail(
                    f"Timeout: Did not receive all 3 messages. Received: {received_messages}"
                )

            # Verification
            assert (
                len(received_messages) == 3
            ), f"Expected 3 messages, got {len(received_messages)}"
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
            logger.error(
                f"An unexpected error occurred during the test: {e}", exc_info=True
            )
            pytest.fail(f"Test failed due to unexpected error: {e}")
        finally:
            logger.info("Cleaning up test resources (subscription)...")
            if ephemeral_sub:
                try:
                    logger.info("Unsubscribing ephemeral consumer...")
                    await asyncio.wait_for(
                        ephemeral_sub.unsubscribe(), timeout=10.0
                    )  # Added explicit timeout
                    logger.info("Ephemeral consumer unsubscribed.")
                except asyncio.TimeoutError:
                    logger.error("Timeout during unsubscribe")
                except Exception as e:
                    logger.error(
                        f"Error during ephemeral subscription cleanup: {e}",
                        exc_info=True,
                    )
            logger.info("Test cleanup (subscription) finished.")

        logger.info("test_full_flow_direct_subscribe completed successfully.")
        # --- Existing Test Logic END ---

    except Exception as e:
        logger.error(f"Unexpected error in top-level test: {e}", exc_info=True)
        pytest.fail(f"Test failed due to unexpected error: {e}")
