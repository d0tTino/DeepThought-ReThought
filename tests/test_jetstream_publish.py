import asyncio
import nats
import pytest
import logging
import uuid
from nats.js.client import JetStreamContext

# Basic logging for the test
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TEST_SUBJECT = f"dtr.test.publish.{uuid.uuid4()}" # Publish to a subject within the stream
TEST_PAYLOAD = b"JetStream test publish"
STREAM_NAME = "deepthought_events" # Ensure this matches the stream created by setup_jetstream.py

@pytest.mark.asyncio
async def test_nats_jetstream_publish_only():
    """Tests only publishing a message to JetStream."""
    nc = None

    try:
        # Connect Client and get JetStream context
        logger.info("Connecting NATS client...")
        nc = await nats.connect("nats://localhost:4222", name="pytest_js_pub_only")
        assert nc.is_connected
        logger.info("NATS client connected.")

        logger.info("Getting JetStream context...")
        js = nc.jetstream(timeout=5.0) # Add timeout for context acquisition
        assert js is not None
        logger.info("JetStream context obtained.")

        # Check if stream exists (optional but good practice)
        try:
             stream_info = await js.stream_info(STREAM_NAME)
             logger.info(f"Confirmed stream '{STREAM_NAME}' exists.")
             assert stream_info.config.name == STREAM_NAME
        except Exception as e:
             logger.error(f"Failed to confirm stream '{STREAM_NAME}' exists: {e}. Ensure setup_jetstream.py ran.")
             pytest.fail(f"JetStream stream '{STREAM_NAME}' not found or accessible.")

        # Publish to JetStream
        logger.info(f"Attempting JetStream publish to subject: {TEST_SUBJECT}")
        # Add a timeout to the publish call itself
        ack = await js.publish(TEST_SUBJECT, TEST_PAYLOAD, timeout=5.0) 
        assert ack is not None
        assert ack.seq > 0 # Check if we got a valid sequence number
        logger.info(f"JetStream publish successful. Ack: stream={ack.stream}, seq={ack.seq}")

    except asyncio.TimeoutError:
        logger.error("Timeout during JetStream operation (connect, context, or publish).")
        pytest.fail("Timeout during JetStream operation.")
    except Exception as e:
        logger.error(f"An error occurred during the JetStream publish test: {e}")
        pytest.fail(f"An error occurred: {e}")
    finally:
        # Cleanup
        logger.info("Cleaning up connection...")
        if nc and nc.is_connected:
            await nc.close()
            logger.info("NATS client closed.") 
