class TemplateServiceSubscriber:
    """Example subscriber using shared NATS connection."""

    def __init__(self, nats_client, js_context):
        from deepthought.eda.subscriber import Subscriber

        self._subscriber = Subscriber(nats_client, js_context)

    async def start(self, subject: str) -> None:
        await self._subscriber.subscribe(subject, self._handle, use_jetstream=True, durable="template")

    async def _handle(self, msg):
        await msg.ack()
