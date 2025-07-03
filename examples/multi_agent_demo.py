import asyncio
import logging
import os
import uuid

import nats
from nats.js.api import DiscardPolicy, RetentionPolicy, StorageType, StreamConfig

from deepthought.modules import (
    BasicLLM,
    BasicMemory,
    InputHandler,
    LLMStub,
    OutputHandler,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

STREAM_NAME = "deepthought_events"


async def ensure_stream(js):
    try:
        await js.stream_info(STREAM_NAME)
    except Exception:
        config = StreamConfig(
            name=STREAM_NAME,
            subjects=["dtr.>"],
            retention=RetentionPolicy.LIMITS,
            storage=StorageType.MEMORY,
            max_msgs_per_subject=100,
            discard=DiscardPolicy.OLD,
        )
        await js.add_stream(config)


async def main() -> None:
    url = os.getenv("NATS_URL", "nats://localhost:4222")
    nc = await nats.connect(url)
    js = nc.jetstream()

    await ensure_stream(js)

    memory = BasicMemory(nc, js)
    try:
        llm = BasicLLM(nc, js)
    except ImportError:
        logger.warning("BasicLLM dependencies missing; falling back to LLMStub")
        llm = LLMStub(nc, js)

    input_handlers = [InputHandler(nc, js) for _ in range(3)]
    done = asyncio.Event()
    msg_count = 0

    def make_callback(idx: int):
        def cb(input_id: str, text: str) -> None:
            nonlocal msg_count
            logger.info("Agent %s says: %s", idx + 1, text)
            msg_count += 1
            next_idx = (idx + 1) % 3
            if msg_count >= 3:
                done.set()
            else:
                asyncio.create_task(input_handlers[next_idx].process_input(text))

        return cb

    output_handlers = [OutputHandler(nc, js, output_callback=make_callback(i)) for i in range(3)]

    await asyncio.gather(
        memory.start_listening(durable_name=f"mem_demo_{uuid.uuid4()}"),
        llm.start_listening(durable_name=f"llm_demo_{uuid.uuid4()}"),
        *(oh.start_listening(durable_name=f"out_demo_{i}_{uuid.uuid4()}") for i, oh in enumerate(output_handlers)),
    )

    await asyncio.sleep(1.0)
    await input_handlers[0].process_input("Hello from agent 1!")

    await done.wait()

    await memory.stop_listening()
    await llm.stop_listening()
    await asyncio.gather(*(oh.stop_listening() for oh in output_handlers))
    await nc.drain()


if __name__ == "__main__":
    asyncio.run(main())
