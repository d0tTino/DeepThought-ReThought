import asyncio
import logging

from nats.aio.client import Client as NATS

from deepthought.config import get_settings
from deepthought.graph import GraphConnector, GraphDAL
from deepthought.modules import KnowledgeGraphMemory

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def main() -> None:
    settings = get_settings()
    nc = NATS()
    await nc.connect(servers=[settings.nats_url])
    js = nc.jetstream()

    connector = GraphConnector(
        host=settings.mg_host,
        port=settings.mg_port,
        username=settings.mg_user,
        password=settings.mg_password,
    )
    dal = GraphDAL(connector)
    memory = KnowledgeGraphMemory(nc, js, dal)
    await memory.start_listening()
    logger.info("KnowledgeGraphMemory listening for INPUT_RECEIVED events")
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
