import asyncio
import nats
import pytest
import logging
import uuid
from src.deepthought.config import DEFAULT_CONFIG

# Basic logging for the test
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

TEST_SUBJECT = f"test.basic_communication.{uuid.uuid4()}"
TEST_PAYLOAD = b"ping"


@pytest.mark.asyncio
async def test_nats_basic_pub_sub():
    """Tests basic NATS publish and subscribe without JetStream."""
    nc_pub = None
    nc_sub = None
    sub = None
    received_message = None
    received_event = asyncio.Event()

    async def message_handler(msg):
        nonlocal received_message
        logger.info(f"Received message on subject '{msg.subject}': {msg.data.decode()}")
        received_message = msg.data
        received_event.set()

    try:
        # Connect Publisher Client
        logger.info("Connecting publisher client...")
        nc_pub = await nats.connect(DEFAULT_CONFIG.nats_url, name="pytest_basic_pub")
        assert nc_pub.is_connected
        logger.info("Publisher client connected.")

        # Connect Subscriber Client
        logger.info("Connecting subscriber client...")
        nc_sub = await nats.connect(DEFAULT_CONFIG.nats_url, name="pytest_basic_sub")
        assert nc_sub.is_connected
        logger.info("Subscriber client connected.")

        # Subscribe
        logger.info(f"Subscribing to subject: {TEST_SUBJECT}")
        sub = await nc_sub.subscribe(TEST_SUBJECT, cb=message_handler)
        await asyncio.sleep(0.1)  # Allow subscription to register

        # Publish
        logger.info(f"Publishing message to subject: {TEST_SUBJECT}")
        await nc_pub.publish(TEST_SUBJECT, TEST_PAYLOAD)
        logger.info("Message published.")

        # Wait for message
        logger.info("Waiting for message...")
        await asyncio.wait_for(received_event.wait(), timeout=5.0)

        # Assertions
        assert received_message is not None, "Did not receive any message"
        assert (
            received_message == TEST_PAYLOAD
        ), f"Received payload '{received_message.decode()}' does not match expected '{TEST_PAYLOAD.decode()}'"
        logger.info("Message received and verified successfully.")

    except asyncio.TimeoutError:
        logger.error("Timeout waiting for message.")
        pytest.fail("Timeout waiting for message in basic pub/sub test.")
    except Exception as e:
        logger.error(f"An error occurred during the test: {e}")
        pytest.fail(f"An error occurred: {e}")
    finally:
        # Cleanup
        logger.info("Cleaning up connections...")
        if sub:
            try:
                await sub.unsubscribe()
                logger.info("Unsubscribed.")
            except Exception as e:
                logger.warning(f"Error unsubscribing: {e}")
        if nc_pub and nc_pub.is_connected:
            await nc_pub.close()
            logger.info("Publisher client closed.")
        if nc_sub and nc_sub.is_connected:
            await nc_sub.close()
            logger.info("Subscriber client closed.")
