"""
Simple test script for NATS communication.
"""

import asyncio
import json
import uuid
import datetime

import nats

async def run_test():
    print("Connecting to NATS server...")
    nc = await nats.connect("nats://localhost:4222")
    
    try:
        # Test basic pub/sub
        test_subject = f"test.{uuid.uuid4()}"
        print(f"Using test subject: {test_subject}")
        
        # Create a future to wait for the message
        received_future = asyncio.Future()
        
        # Subscribe to our test subject
        async def message_handler(msg):
            data_str = msg.data.decode()
            data = json.loads(data_str)
            print(f"Received message: {data}")
            received_future.set_result(data)
        
        sub = await nc.subscribe(test_subject, cb=message_handler)
        print(f"Subscribed to {test_subject}")
        
        # Publish a test message
        test_data = {
            "test_id": str(uuid.uuid4()),
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "message": "Hello NATS!"
        }
        
        print(f"Publishing message: {test_data}")
        await nc.publish(test_subject, json.dumps(test_data).encode())
        
        # Wait for the message or timeout
        try:
            print("Waiting for message...")
            received = await asyncio.wait_for(received_future, timeout=5.0)
            print(f"Test successful! Received: {received}")
            if received["test_id"] == test_data["test_id"]:
                print("Message ID match confirmed.")
            else:
                print(f"ERROR: Message ID mismatch! Expected {test_data['test_id']}, got {received['test_id']}")
        except asyncio.TimeoutError:
            print("TIMEOUT: Did not receive the test message within 5 seconds.")
            
        # Cleanup
        await sub.unsubscribe()
        print("Unsubscribed.")
        
    finally:
        # Close the connection
        await nc.drain()
        print("Connection closed.")

if __name__ == "__main__":
    print("Running NATS simple test...")
    asyncio.run(run_test()) 