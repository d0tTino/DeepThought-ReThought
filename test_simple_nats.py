import os
import pytest

if os.getenv("RUN_NATS_TESTS") != "1":
    pytest.skip("NATS tests skipped (set RUN_NATS_TESTS=1 to enable)", allow_module_level=True)

import asyncio
from nats.aio.client import Client as NATS
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def run():
    # Connect to NATS
    logger.info("Attempting to connect to NATS at nats://localhost:4222")
    nc = NATS()
    
    try:
        await nc.connect(servers=["nats://localhost:4222"], connect_timeout=10)
        if nc.is_connected:
            logger.info("NATS connection successful.")
            
            # Get JetStream context
            logger.info("Getting JetStream context...")
            js = nc.jetstream(timeout=5.0)  # Shorter timeout
            logger.info("JetStream context obtained.")
            
            try:
                # Try to get stream info or create a stream
                stream_name = "TEST_STREAM"
                logger.info(f"Checking if stream '{stream_name}' exists...")
                
                try:
                    stream_info = await js.stream_info(stream_name)
                    logger.info(f"Stream '{stream_name}' exists: {stream_info}")
                except Exception as e:
                    logger.info(f"Stream does not exist, creating it: {e}")
                    
                    # Create stream
                    from nats.js.api import StreamConfig, RetentionPolicy, StorageType, DiscardPolicy
                    stream_config = StreamConfig(
                        name=stream_name,
                        subjects=["test.>"],
                        retention=RetentionPolicy.LIMITS,
                        storage=StorageType.MEMORY,  # Use memory for test
                        max_msgs_per_subject=10,
                        discard=DiscardPolicy.OLD,
                    )
                    
                    stream = await js.add_stream(stream_config)
                    logger.info(f"Stream created: {stream}")
                
                # Publish a message
                await js.publish("test.message", b"Hello World!")
                logger.info("Message published.")
                
            except Exception as e:
                logger.error(f"Error with JetStream operations: {e}")
        
    except Exception as e:
        logger.error(f"Error connecting to NATS: {e}")
    
    finally:
        # Clean up
        if nc.is_connected:
            logger.info("Closing NATS connection.")
            await nc.close()
            logger.info("NATS connection closed.")

if __name__ == "__main__":
    asyncio.run(run()) 