import asyncio
import logging
import os
import pytest
import pytest_asyncio
import subprocess
import sys
import time # For unique IDs
import uuid

# Add the src directory to the path for imports if not already configured in pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nats.aio.client import Client as NATS
from nats.js import JetStreamContext

# Import configuration from src
from src.deepthought.config import get_nats_url, get_nats_stream_name
from src.deepthought.modules import InputHandler, MemoryStub, LLMStub, OutputHandler
from src.deepthought.eda.events import EventSubjects # Assuming this defines the subjects used by modules

# Configure logging for the test
logger = logging.getLogger(__name__)

# --- Configuration for E2E Test ---
# NATS_URL is now fetched by get_nats_url()
E2E_CLIENT_PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "e2e_csharp_client"))
# Determine executable path based on OS - adjust if needed, especially for Windows
E2E_CLIENT_EXECUTABLE_NAME = "E2ETestClient"
E2E_CLIENT_EXE_PATH = os.path.join(E2E_CLIENT_PROJECT_PATH, "bin", "Debug", "net6.0", E2E_CLIENT_EXECUTABLE_NAME + (".exe" if os.name == 'nt' else ""))

# Define subjects for the E2E flow
# These should align with what InputHandler listens to and OutputHandler publishes on
TEST_INPUT_SUBJECT = EventSubjects.INPUT_RECEIVED # Or a more specific E2E test subject if desired
TEST_RESPONSE_SUBJECT_PREFIX = "e2e.response" # C# client will listen on TEST_RESPONSE_SUBJECT_PREFIX.<unique_id>

# Helper function to ensure the JetStream stream exists (adapted from test_module_integration)
async def ensure_stream_exists(js: JetStreamContext, stream_name: str, subjects: list[str]):
    try:
        logger.info(f"Checking if stream '{stream_name}' exists...")
        await js.stream_info(stream_name)
        logger.info(f"Stream '{stream_name}' already exists.")
        # Potentially update stream subjects if needed, though not done here
        return True
    except Exception:
        logger.info(f"Stream '{stream_name}' does not exist, creating it...")
        from nats.js.api import StreamConfig, RetentionPolicy, StorageType, DiscardPolicy
        stream_config = StreamConfig(
            name=stream_name,
            subjects=subjects,
            retention=RetentionPolicy.LIMITS,
            storage=StorageType.MEMORY,
            max_msgs_per_subject=100, # Keep low for tests
            discard=DiscardPolicy.OLD,
        )
        try:
            await js.add_stream(stream_config)
            logger.info(f"Stream '{stream_name}' created successfully with subjects: {subjects}.")
            return True
        except Exception as create_err:
            logger.error(f"Failed to create stream '{stream_name}': {create_err}")
            return False

@pytest_asyncio.fixture(scope="module")
async def nats_connection_module():
    logger.info("Module Fixture: Connecting to NATS...")
    nc = NATS()
    try:
        await nc.connect(servers=[get_nats_url()], connect_timeout=10, name="e2e_test_orchestrator") # Use imported config
        logger.info("Module Fixture: NATS connection successful.")

        js = nc.jetstream(timeout=5.0)
        assert js is not None, "Module Fixture: Failed to get JetStream context"

        # Ensure the main stream 'deepthought_events' with 'dtr.>' exists for modules
        # The modules use subjects like dtr.input.received, dtr.memory.retrieved etc.
        # EventSubjects.INPUT_RECEIVED is 'dtr.input.received'
        # EventSubjects.RESPONSE_GENERATED is 'dtr.llm.response_generated'
        # OutputHandler listens to EventSubjects.RESPONSE_GENERATED and publishes to a subject provided in the incoming message's reply_to_e2e field.
        # Using get_nats_stream_name() for stream name from config.
        assert await ensure_stream_exists(js, get_nats_stream_name(), ["dtr.>"]), f"Module Fixture: Failed to ensure '{get_nats_stream_name()}' stream"

        yield nc
    finally:
        logger.info("Module Fixture: Closing NATS connection...")
        if nc.is_connected:
            await nc.drain(timeout=5.0)
        logger.info("Module Fixture: NATS connection closed.")


@pytest_asyncio.fixture(scope="module")
async def python_eda_modules(nats_connection_module):
    nc = nats_connection_module
    js = nc.jetstream()

    logger.info("Module Fixture: Initializing Python EDA modules...")
    # The OutputHandler needs a way to know where to send the E2E response.
    # We'll modify it slightly for E2E: if a message has 'reply_to_e2e' in its payload, it uses that.
    # This requires a custom output_callback for the OutputHandler in this test.

    # This dictionary will store final responses received by our custom callback
    e2e_test_responses = {}
    e2e_response_events = {} # Stores asyncio.Event per input_id

    def e2e_output_callback(input_id, response_data, original_message_payload):
        logger.info(f"[E2E Callback] Output for input_id {input_id}: {response_data}")
        e2e_test_responses[input_id] = response_data
        if input_id in e2e_response_events:
            e2e_response_events[input_id].set()

        # For the E2E test, the C# client needs the response on a specific subject.
        # The OutputHandler is modified to look for `reply_to_e2e` in the original_message_payload
        # This field will be set by the InputHandler when it processes the message from the C# client.
        reply_subject_for_e2e_client = original_message_payload.get("reply_to_e2e")
        if reply_subject_for_e2e_client:
            async def publish_to_e2e_client():
                try:
                    # We need to serialize the response_data before publishing
                    payload_bytes = str(response_data).encode('utf-8') # Or JSON, ensure C# client expects this
                    await js.publish(reply_subject_for_e2e_client, payload_bytes)
                    logger.info(f"[E2E Callback] Forwarded response for {input_id} to E2E client at {reply_subject_for_e2e_client}")
                except Exception as e:
                    logger.error(f"[E2E Callback] Error forwarding response to {reply_subject_for_e2e_client}: {e}")
            asyncio.create_task(publish_to_e2e_client())


    input_handler = InputHandler(nc, js) # Will publish to EventSubjects.INPUT_RECEIVED ('dtr.input.received')
    memory_stub = MemoryStub(nc, js)     # Listens to INPUT_RECEIVED, publishes to EventSubjects.MEMORY_RETRIEVED ('dtr.memory.retrieved')
    llm_stub = LLMStub(nc, js)           # Listens to MEMORY_RETRIEVED, publishes to EventSubjects.RESPONSE_GENERATED ('dtr.llm.response_generated')
    output_handler = OutputHandler(nc, js, output_callback=e2e_output_callback) # Listens to RESPONSE_GENERATED

    modules = {
        "input_handler": input_handler, # Not started here, as it only publishes
        "memory_stub": memory_stub,
        "llm_stub": llm_stub,
        "output_handler": output_handler,
        "e2e_test_responses": e2e_test_responses, # For assertions
        "e2e_response_events": e2e_response_events
    }

    logger.info("Module Fixture: Starting Python EDA module listeners...")
    await asyncio.gather(
        memory_stub.start_listening(durable_name="e2e_mem_listener"),
        llm_stub.start_listening(durable_name="e2e_llm_listener"),
        output_handler.start_listening(durable_name="e2e_out_listener")
    )
    logger.info("Module Fixture: Python EDA module listeners started.")

    yield modules

    logger.info("Module Fixture: Stopping Python EDA module listeners...")
    await asyncio.gather(
        memory_stub.stop_listening(),
        llm_stub.stop_listening(),
        output_handler.stop_listening()
    )
    logger.info("Module Fixture: Python EDA module listeners stopped.")


@pytest.mark.asyncio
async def test_full_e2e_flow(python_eda_modules):
    """
    Tests the full end-to-end flow: C# Client -> Python EDA -> C# Client
    """
    if not os.path.exists(E2E_CLIENT_EXE_PATH):
        pytest.fail(f"C# E2E client executable not found at {E2E_CLIENT_EXE_PATH}. Build it first using 'dotnet build {E2E_CLIENT_PROJECT_PATH}'.")

    input_handler = python_eda_modules["input_handler"]
    e2e_test_responses = python_eda_modules["e2e_test_responses"]
    e2e_response_events = python_eda_modules["e2e_response_events"]

    # Unique ID for this test run to avoid subject collisions if tests run in parallel (though not typical for E2E)
    # And to ensure C# client gets its own response.
    test_run_id = str(uuid.uuid4())

    # Subject C# client will publish to (InputHandler will receive this)
    # The InputHandler needs to be aware of this or listen broadly on dtr.input.e2e.*
    # For simplicity, let's use the existing EventSubjects.INPUT_RECEIVED
    # The payload from C# will include the test_run_id and the specific reply subject.
    csharp_publish_subject = EventSubjects.INPUT_RECEIVED # 'dtr.input.received'

    # Subject C# client will subscribe to for the final response from OutputHandler
    csharp_subscribe_subject = f"{TEST_RESPONSE_SUBJECT_PREFIX}.{test_run_id}"

    test_message_payload_from_csharp = f"E2E test message from C# client, run ID: {test_run_id}"

    # The Python InputHandler needs to pass `csharp_subscribe_subject` to the OutputHandler.
    # We achieve this by having InputHandler add a `reply_to_e2e` field to the NATS message it publishes.
    # This requires a slight modification or configuration in InputHandler if it's not already generic enough.
    # For this test, we assume InputHandler will see a dict payload and pass it through.
    # The C# client will send a JSON string representing a dictionary.
    csharp_payload_dict = {
        "text_input": test_message_payload_from_csharp,
        "input_id": test_run_id, # C# client creates its own input_id
        "reply_to_e2e": csharp_subscribe_subject # This is key for routing response back to C#
    }
    import json
    csharp_payload_json = json.dumps(csharp_payload_dict)


    command = [
        E2E_CLIENT_EXE_PATH,
        get_nats_url(), # Use imported config
        csharp_publish_subject,
        csharp_payload_json, # Send JSON string
        csharp_subscribe_subject,
        "15000"  # Timeout for C# client in milliseconds (e.g., 15 seconds)
    ]

    logger.info(f"Starting C# E2E client: {' '.join(command)}")
    process = None
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Wait for the C# process to complete, with a timeout
        # This timeout should be longer than the C# client's internal timeout
        stdout, stderr = process.communicate(timeout=20.0)
        return_code = process.returncode

        logger.info(f"C# Client STDOUT:\n{stdout}")
        if stderr:
            logger.error(f"C# Client STDERR:\n{stderr}")

        assert return_code == 0, f"C# client exited with error code {return_code}. STDERR: {stderr}"

        # Verify the C# client received a response (from its stdout)
        # The C# client prints the received message to its stdout.
        # Example: "Success: Response received." followed by the actual message on a new line.
        assert "Success: Response received." in stdout, "C# client did not report successful response receipt."

        # The actual response from the Python EDA will be on the next line in stdout
        # Example: "LLM response to: Memory content for: E2E test message..."
        # We can make this assertion more specific if we know the exact expected Python output format.
        # For now, just check that *some* response payload was printed by C# client.
        lines = stdout.strip().split('\n')
        actual_response_payload_from_csharp_stdout = lines[-1] # Last line should be the payload

        logger.info(f"Response payload printed by C# client: {actual_response_payload_from_csharp_stdout}")
        assert "LLM response to: Memory content for:" in actual_response_payload_from_csharp_stdout, "Response payload from C# stdout does not match expected pattern."

        # Also, check if the Python side (via e2e_output_callback) recorded the response for this ID.
        # This is a secondary check, primary is C# client success.
        # Note: The input_id used by Python modules will be generated by InputHandler.
        # The test_run_id here is for correlating the C# client's request.
        # The `e2e_output_callback` uses the `input_id` generated by `InputHandler`.
        # The `original_message_payload` in `e2e_output_callback` will be the `csharp_payload_dict`.
        # So the `reply_to_e2e` forwarding works.

        # To verify the Python side, we need to wait for the event related to *an* input_id
        # that corresponds to this E2E test. This is tricky because the C# client
        # doesn't know the Python InputHandler's generated input_id.
        # However, the `e2e_output_callback` *does* log and forward.
        # The C# client's success is the primary assertion for the E2E flow.

        logger.info(f"E2E test for run ID {test_run_id} completed successfully.")

    except subprocess.TimeoutExpired:
        logger.error("Python test timed out waiting for C# client.")
        if process:
            process.kill()
            stdout, stderr = process.communicate()
            logger.error(f"C# Client STDOUT on kill:\n{stdout}")
            logger.error(f"C# Client STDERR on kill:\n{stderr}")
        pytest.fail("Python test timed out waiting for C# client.")
    except Exception as e:
        logger.error(f"An error occurred during E2E test: {e}")
        if process and process.poll() is None: # Check if process is still running
             process.kill()
        pytest.fail(f"An error occurred during E2E test: {e}")
    finally:
        if process and process.poll() is None: # Ensure process is killed if it's still running
            logger.info("Killing C# subprocess in finally block...")
            process.kill()
            process.wait() # Wait for the process to actually terminate
            logger.info("C# subprocess killed.")
