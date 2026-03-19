from __future__ import annotations

import logging

from nats.aio.client import Client as NATS
from nats.aio.msg import Msg
from nats.js.client import JetStreamContext

from deepthought.eda import Publisher, Subscriber
from deepthought.eda.contracts import EventEnvelope

logger = logging.getLogger(__name__)


class DemoService:
    """Skeleton service using Publisher and Subscriber."""

    def __init__(self, nats_client: NATS, js_context: JetStreamContext) -> None:
        self._publisher = Publisher(nats_client, js_context)
        self._subscriber = Subscriber(nats_client, js_context)

    async def _handle_input(self, msg: Msg) -> None:
        logger.info("Handling message %s", msg.subject)
        try:
            envelope = EventEnvelope.build(
                subject="dtr.template.output",
                payload={"data": msg.data.decode(errors="replace")},
                producer=self.__class__.__name__,
            )
            await self._publisher.publish("dtr.template.output", envelope.__dict__, use_jetstream=True)
        finally:
            await msg.ack()

    async def start(self, durable_name: str = "demo_listener") -> bool:
        await self._subscriber.subscribe(
            subject="dtr.template.input",
            handler=self._handle_input,
            use_jetstream=True,
            durable=durable_name,
        )
        return True

    async def stop(self) -> None:
        await self._subscriber.unsubscribe_all()
