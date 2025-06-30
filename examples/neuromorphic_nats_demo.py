"""Example showing neuromorphic processing with NATS events."""

import asyncio
import json
import logging
import uuid

import nats

from deepthought.eda.publisher import Publisher
from deepthought.eda.subscriber import Subscriber
from deepthought.neuromorphic import NeuromorphicProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def main() -> None:
    nc = await nats.connect("nats://localhost:4222")
    js = nc.jetstream()

    processor = NeuromorphicProcessor()
    publisher = Publisher(nc, js)
    subscriber = Subscriber(nc, js)

    async def handle_input(msg: nats.aio.msg.Msg) -> None:
        data = json.loads(msg.data.decode())
        value = float(data.get("value", 0))
        result = processor.run(value)
        payload = {"result": result, "input_id": data.get("input_id", str(uuid.uuid4()))}
        await publisher.publish("dtr.neuro.output", payload, use_jetstream=True)
        await msg.ack()
        logger.info("Processed value %s -> %s", value, result)

    await subscriber.subscribe(
        "dtr.neuro.input",
        handler=handle_input,
        use_jetstream=True,
        durable="neuromorphic_demo",
    )

    logger.info("Waiting for dtr.neuro.input events. Press Ctrl+C to exit.")
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    await subscriber.unsubscribe_all()
    await nc.drain()


if __name__ == "__main__":
    asyncio.run(main())
