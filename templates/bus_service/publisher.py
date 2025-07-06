class TemplateServicePublisher:
    """Example publisher using JetStream persistence."""

    def __init__(self, nats_client, js_context):
        from deepthought.eda.publisher import Publisher

        self._publisher = Publisher(nats_client, js_context)

    async def publish_example(self, subject: str, data: str) -> None:
        await self._publisher.publish(subject, data, use_jetstream=True)
