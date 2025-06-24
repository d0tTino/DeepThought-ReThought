import asyncio
import logging
import os

from nats.aio.client import Client as NATS

from deepthought.graph import GraphConnector, GraphDAL
from deepthought.modules import KnowledgeGraphMemory

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def main() -> None:
    nc = NATS()
    await nc.connect(servers=[os.getenv("NATS_URL", "nats://localhost:4222")])
    js = nc.jetstream()

    connector = GraphConnector(
        host=os.getenv("MG_HOST", "localhost"),
        port=int(os.getenv("MG_PORT", "7687")),
        username=os.getenv("MG_USER", ""),
        password=os.getenv("MG_PASSWORD", ""),
    )
    dal = GraphDAL(connector)
    memory = KnowledgeGraphMemory(nc, js, dal)
    await memory.start_listening()
    logger.info("KnowledgeGraphMemory listening for INPUT_RECEIVED events")
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
