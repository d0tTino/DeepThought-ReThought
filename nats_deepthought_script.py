"""
Test script for DeepThought EDA subjects communication.
"""

import asyncio
import json
import uuid
import datetime
import sys
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the src directory to the path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

import nats
from deepthought.eda.events import EventSubjects

# Print the subjects to make sure they're loaded correctly
print(f"INPUT_RECEIVED subject: {EventSubjects.INPUT_RECEIVED}")
print(f"MEMORY_RETRIEVED subject: {EventSubjects.MEMORY_RETRIEVED}")
print(f"RESPONSE_GENERATED subject: {EventSubjects.RESPONSE_GENERATED}")

async def test_subject(nc, subject, payload):
    """Test sending and receiving on a specific subject."""
    print(f"\n{'='*60}")
    print(f"Testing subject: {subject}")
    print(f"{'='*60}")
    
    # Create a future to wait for the message
    received_future = asyncio.Future()
    
    # Subscribe to the subject
    async def message_handler(msg):
        print(f"MESSAGE HANDLER CALLED for {msg.subject}")
        data_str = msg.data.decode()
        print(f"Raw message data: {data_str}")
        data = json.loads(data_str)
        print(f"Received message on {msg.subject}: {data}")
        received_future.set_result(data)
    
    sub = await nc.subscribe(subject, cb=message_handler)
    print(f"Subscribed to {subject}")
    
    # Publish a test message
    message_data = json.dumps(payload).encode()
    print(f"Encoded message length: {len(message_data)} bytes")
    print(f"Publishing message to {subject}: {payload}")
    await nc.publish(subject, message_data)
    print(f"Message published to {subject}")
    
    # Wait for the message or timeout
    try:
        print(f"Waiting for message on {subject}...")
        received = await asyncio.wait_for(received_future, timeout=5.0)
        print(f"Success! Received on {subject}: {received}")
        if received["input_id"] == payload["input_id"]:
            print("Message ID match confirmed.")
        else:
            print(f"ERROR: Message ID mismatch! Expected {payload['input_id']}, got {received['input_id']}")
    except asyncio.TimeoutError:
        print(f"TIMEOUT: Did not receive message on {subject} within 5 seconds.")
        received = None
        
    # Cleanup
    await sub.unsubscribe()
    print(f"Unsubscribed from {subject}")
    return received

async def run_test():
    print("Connecting to NATS server...")
    nc = await nats.connect("nats://localhost:4222")
    print(f"Connected to NATS server at {nc.connected_url.netloc}")
    
    try:
        # Print NATS server info
        server_info = nc._server_info
        print(f"Server info: {server_info}")
        print(f"Server ID: {server_info['server_id']}")
        
        # Create a unique input ID for this test
        input_id = str(uuid.uuid4())
        timestamp = datetime.datetime.utcnow().isoformat()
        print(f"Using test input_id: {input_id}")
        
        # Test INPUT_RECEIVED subject
        input_payload = {
            "user_input": "Test input for DeepThought",
            "input_id": input_id,
            "timestamp": timestamp
        }
        print(f"Starting test for INPUT_RECEIVED subject...")
        input_received = await test_subject(nc, EventSubjects.INPUT_RECEIVED, input_payload)
        
        if input_received:
            # Test MEMORY_RETRIEVED subject
            memory_payload = {
                "input_id": input_id,
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "retrieved_knowledge": {
                    "retrieved_knowledge": {
                        "facts": ["This is a test fact", f"User asked: {input_payload['user_input']}"],
                        "source": "memory_stub"
                    }
                }
            }
            print(f"Starting test for MEMORY_RETRIEVED subject...")
            memory_received = await test_subject(nc, EventSubjects.MEMORY_RETRIEVED, memory_payload)
            
            if memory_received:
                # Test RESPONSE_GENERATED subject
                response_payload = {
                    "input_id": input_id,
                    "timestamp": datetime.datetime.utcnow().isoformat(),
                    "final_response": f"This is a test response for: {input_payload['user_input']}",
                    "confidence": 0.95
                }
                print(f"Starting test for RESPONSE_GENERATED subject...")
                await test_subject(nc, EventSubjects.RESPONSE_GENERATED, response_payload)
        
    finally:
        # Close the connection
        print("Closing NATS connection...")
        await nc.drain()
        print("\nConnection closed.")

if __name__ == "__main__":
    print("Running DeepThought EDA subjects test...")
    try:
        asyncio.run(run_test())
    except Exception as e:
        print(f"ERROR in test execution: {e}")
        import traceback
        traceback.print_exc() 
