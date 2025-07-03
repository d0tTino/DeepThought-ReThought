from __future__ import annotations

import logging

from nats.aio.client import Client as NATS
from nats.aio.msg import Msg
from nats.js.client import JetStreamContext

from deepthought.eda import Publisher, Subscriber

logger = logging.getLogger(__name__)


class TemplateService:
    """Skeleton service using Publisher and Subscriber."""

    def __init__(self, nats_client: NATS, js_context: JetStreamContext) -> None:
        self._publisher = Publisher(nats_client, js_context)
        self._subscriber = Subscriber(nats_client, js_context)

    async def _handle_input(self, msg: Msg) -> None:
        logger.info("Handling message %s", msg.subject)
        # TODO: add processing logic
        await msg.ack()

    async def start(self, durable_name: str = "template_service_listener") -> bool:
        await self._subscriber.subscribe(
            subject="dtr.template.input",
            handler=self._handle_input,
            use_jetstream=True,
            durable=durable_name,
        )
        return True

    async def stop(self) -> None:
        await self._subscriber.unsubscribe_all()
