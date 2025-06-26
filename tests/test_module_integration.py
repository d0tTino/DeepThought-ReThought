# File: tests/test_module_integration.py
"""
Integration test for DeepThought reThought system modules.
Tests the full event flow between basic functional modules.
"""
import asyncio
import json
import logging
import os
import sys
import tempfile

import pytest
from nats.aio.client import Client as NATS
from nats.js import JetStreamContext
from nats.js.api import DiscardPolicy, RetentionPolicy, StorageType, StreamConfig

from tests.helpers import nats_server_available

# Add the src directory to the path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the modules to test
from src.deepthought.modules import (
    BasicLLM,
    GraphMemory,
    InputHandler,
    LLMStub,
    OutputHandler,
    ProductionLLM,
)


class DummyMemory:
    def __init__(self):
        self.prompt = None

    def retrieve_context(self, prompt: str):
        self.prompt = prompt
        return ["fact1", "fact2"]


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Helper function to get NATS URL from environment variable
def get_nats_url() -> str:
    return os.getenv("NATS_URL", "nats://localhost:4222")


# Stream name - using the same as in setup_jetstream.py
STREAM_NAME = "deepthought_events"

# Temporary file used for GraphMemory integration tests
GRAPH_MEMORY_FILE = tempfile.NamedTemporaryFile(delete=False, suffix="_graph.json").name


# Helper function to ensure the JetStream stream exists
async def ensure_stream_exists(js: JetStreamContext, stream_name: str) -> bool:
    """Ensure the JetStream stream exists and has the correct configuration."""
    try:
        # Check if the stream exists
        logger.info(f"Checking if stream '{stream_name}' exists...")
        await js.stream_info(stream_name)
        logger.info(f"Stream '{stream_name}' already exists.")
        return True
    except Exception as e:
        logger.info(f"Stream '{stream_name}' does not exist, creating it... ({e})")

        # Create the stream with appropriate settings
        stream_config = StreamConfig(
            name=stream_name,
            subjects=["dtr.>"],  # All DeepThought subjects
            retention=RetentionPolicy.LIMITS,
            storage=StorageType.MEMORY,  # Use memory storage for tests
            max_msgs_per_subject=100,
            discard=DiscardPolicy.OLD,
        )

        try:
            await js.add_stream(stream_config)
            logger.info(f"Stream '{stream_name}' created successfully.")
            return True
        except Exception as create_err:
            logger.error(f"Failed to create stream '{stream_name}': {create_err}")
            return False


# The integration test function
@pytest.mark.asyncio
async def test_full_module_flow():
    """
    Test the full event flow using hierarchical memory via InputHandler.
    1. Input -> InputHandler publishes INPUT_RECEIVED
    2. InputHandler fetches context and publishes MEMORY_RETRIEVED
    3. BasicLLM subscribes to MEMORY_RETRIEVED, publishes RESPONSE_GENERATED
    4. OutputHandler subscribes to RESPONSE_GENERATED, handles the final output
    """
    if not nats_server_available(get_nats_url()):
        pytest.skip("NATS server not available")
    nc = None
    llm_module = None
    output_handler = None

    try:
        # --- Connect NATS ---
        logger.info(f"Attempting to connect to NATS at {get_nats_url()}")
        nc = NATS()
        await nc.connect(servers=[get_nats_url()], connect_timeout=10)
        if not nc.is_connected:
            pytest.fail("NATS connection failed")
        logger.info("NATS connection successful.")

        # --- Get JetStream context ---
        logger.info("Getting JetStream context...")
        js = nc.jetstream(timeout=30.0)
        if not js:
            pytest.fail("Failed to get JetStream context.")
        logger.info("JetStream context obtained.")

        # --- Ensure stream exists ---
        if not await ensure_stream_exists(js, STREAM_NAME):
            pytest.fail(f"Failed to ensure stream '{STREAM_NAME}' exists.")

        # --- Create event for completion signaling ---
        final_response_received_event = asyncio.Event()
        responses = {}
        test_input_id = None

        # --- Define output callback ---
        def output_callback(input_id, response):
            logger.info(f"Output callback received response for input_id={input_id}: {response}")
            responses[input_id] = response
            # Only set event if it matches the ID we sent for this test run
            if input_id == test_input_id:
                logger.info(f"Correct final response received via callback for ID {input_id}. Setting event.")
                final_response_received_event.set()
            else:
                logger.warning(f"Callback received response for unexpected ID {input_id}, expected {test_input_id}")

        # --- Instantiate module stubs ---
        logger.info("Initializing modules...")
        memory_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json").name
        if os.path.exists(memory_file):
            os.remove(memory_file)
        memory_service = DummyMemory()
        input_handler = InputHandler(nc, js, memory=memory_service)
        llm_cls = ProductionLLM if os.path.isdir("./results/lora-adapter") else BasicLLM
        try:
            llm_module = llm_cls(nc, js)
        except ImportError:
            logger.warning("LLM dependencies missing; falling back to LLMStub")
            llm_module = LLMStub(nc, js)

        output_handler = OutputHandler(nc, js, output_callback=output_callback)
        logger.info("Modules initialized.")

        # --- Set up subscriptions for the modules ---
        logger.info("Starting listeners...")
        results = await asyncio.gather(
            llm_module.start_listening(durable_name="test_llm_listener"),
            output_handler.start_listening(durable_name="test_out_listener"),
            return_exceptions=True,  # Capture exceptions instead of raising immediately
        )

        # Check if any listeners failed to start
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to start listener {i}: {result}")
                pytest.fail(f"Failed to start a required stub listener: {result}")
            elif result is False:  # Check boolean return value
                logger.error(f"Listener {i} reported failure to start.")
                pytest.fail(f"Listener {i} reported failure to start.")

        logger.info("Listeners started successfully.")

        # Wait briefly for all subscriptions to be established
        await asyncio.sleep(1.0)

        # --- Trigger the flow ---
        sample_input = "Test the full module integration flow"
        logger.info(f"Processing input: '{sample_input}'")

        # Process the input via InputHandler
        test_input_id = await input_handler.process_input(sample_input)
        logger.info(f"Input processed, ID: {test_input_id}")

        # --- Wait for the complete flow to finish ---
        logger.info("Waiting for complete flow to finish (final response)...")
        try:
            await asyncio.wait_for(final_response_received_event.wait(), timeout=20.0)
            logger.info("Final response received!")
        except asyncio.TimeoutError:
            logger.error("Timeout waiting for final response.")
            pytest.fail("Timeout: Final response was not received within 20 seconds.")

        # --- Assertions ---
        assert final_response_received_event.is_set(), "Final response signal was not received via callback"
        assert test_input_id in responses, f"OutputHandler did not record response for input_id {test_input_id}"
        assert responses[test_input_id] is not None, "Response content is None"

        logger.info("Full module flow test completed successfully.")

    except Exception as e:
        logger.error(f"An unexpected error occurred during the test: {e}", exc_info=True)
        pytest.fail(f"Test failed due to unexpected error: {e}")
    finally:
        # --- Cleanup ---
        logger.info("Cleaning up test resources...")

        # --- Stop stub listeners ---
        logger.info("Stopping listeners...")
        stubs_to_stop = []
        if llm_module:
            stubs_to_stop.append(llm_module.stop_listening())
        if output_handler:
            stubs_to_stop.append(output_handler.stop_listening())

        if stubs_to_stop:
            results = await asyncio.gather(*stubs_to_stop, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(
                        f"Error during stub listener cleanup {i}: {result}", exc_info=False
                    )  # Avoid deep traceback in cleanup
        logger.info("Listeners stopped.")

        if os.path.exists(memory_file):
            os.remove(memory_file)

        # Close NATS connection
        if nc and nc.is_connected:
            logger.info("Closing NATS connection...")
            await nc.drain()  # Using drain() for cleaner shutdown
            logger.info("NATS connection closed.")
        elif nc:
            logger.warning("NATS client existed but was not connected during teardown.")
        else:
            logger.warning("NATS client was not created during setup.")

        logger.info("Test cleanup finished.")


@pytest.mark.asyncio
async def test_full_module_flow_graph_memory():
    """Same as test_full_module_flow but using GraphMemory module."""
    if not nats_server_available(get_nats_url()):
        pytest.skip("NATS server not available")
    nc = None
    memory_module = None
    llm_module = None
    output_handler = None

    try:
        logger.info(f"Attempting to connect to NATS at {get_nats_url()}")
        nc = NATS()
        await nc.connect(servers=[get_nats_url()], connect_timeout=10)
        if not nc.is_connected:
            pytest.fail("NATS connection failed")
        js = nc.jetstream(timeout=30.0)
        if not js:
            pytest.fail("Failed to get JetStream context.")

        if not await ensure_stream_exists(js, STREAM_NAME):
            pytest.fail(f"Failed to ensure stream '{STREAM_NAME}' exists.")

        final_response_received_event = asyncio.Event()
        responses = {}
        test_input_id = None

        def output_callback(input_id, response):
            responses[input_id] = response
            if input_id == test_input_id:
                final_response_received_event.set()

        logger.info("Initializing modules (GraphMemory)...")
        if os.path.exists(GRAPH_MEMORY_FILE):
            os.remove(GRAPH_MEMORY_FILE)
        memory_service = DummyMemory()
        input_handler = InputHandler(nc, js, memory=memory_service)
        memory_module = GraphMemory(nc, js, graph_file=GRAPH_MEMORY_FILE)
        llm_cls = ProductionLLM if os.path.isdir("./results/lora-adapter") else BasicLLM
        try:
            llm_module = llm_cls(nc, js)
        except ImportError:
            logger.warning("LLM dependencies missing; falling back to LLMStub")
            llm_module = LLMStub(nc, js)
        output_handler = OutputHandler(nc, js, output_callback=output_callback)

        results = await asyncio.gather(
            memory_module.start_listening(durable_name="test_graph_mem_listener"),
            llm_module.start_listening(durable_name="test_graph_llm_listener"),
            output_handler.start_listening(durable_name="test_graph_out_listener"),
            return_exceptions=True,
        )

        for i, result in enumerate(results):
            if isinstance(result, Exception) or result is False:
                pytest.fail(f"Listener failed to start: {result}")

        await asyncio.sleep(1.0)

        sample_input = "Test the graph memory flow"
        test_input_id = await input_handler.process_input(sample_input)

        try:
            await asyncio.wait_for(final_response_received_event.wait(), timeout=20.0)
        except asyncio.TimeoutError:
            pytest.fail("Timeout waiting for final response")

        assert final_response_received_event.is_set()
        assert test_input_id in responses

        with open(GRAPH_MEMORY_FILE, "r", encoding="utf-8") as f:
            graph_json = json.load(f)
        assert graph_json

    finally:
        logger.info("Cleaning up graph memory test resources...")
        stubs_to_stop = []
        if memory_module:
            stubs_to_stop.append(memory_module.stop_listening())
        if llm_module:
            stubs_to_stop.append(llm_module.stop_listening())
        if output_handler:
            stubs_to_stop.append(output_handler.stop_listening())
        if stubs_to_stop:
            await asyncio.gather(*stubs_to_stop, return_exceptions=True)
        if os.path.exists(GRAPH_MEMORY_FILE):
            os.remove(GRAPH_MEMORY_FILE)
        if nc and nc.is_connected:
            await nc.drain()
            logger.info("NATS connection closed.")
        elif nc:
            logger.warning("NATS client existed but was not connected during teardown.")
        else:
            logger.warning("NATS client was not created during setup.")
        logger.info("Graph memory test cleanup finished.")
