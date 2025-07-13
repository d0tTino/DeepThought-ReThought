import asyncio
import time
from typing import Any, Awaitable, Callable

from deepthought.metrics.prometheus import INPUT_LATENCY_SECONDS, INPUTS_TOTAL


def rate_limit(
    capacity: int, refill_interval: float
) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
    """Simple token bucket rate limiter."""

    def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        tokens = capacity
        last = time.monotonic()
        lock = asyncio.Lock()

        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            nonlocal tokens, last
            async with lock:
                now = time.monotonic()
                tokens = min(capacity, tokens + (now - last) / refill_interval)
                last = now
                if tokens < 1:
                    await asyncio.sleep((1 - tokens) * refill_interval)
                    now = time.monotonic()
                    tokens = min(capacity, tokens + (now - last) / refill_interval)
                    last = now
                tokens -= 1
            return await func(*args, **kwargs)

        return wrapper

    return decorator


class TemplateServiceSubscriber:
    """Example subscriber using JetStream persistence."""

    def __init__(self, nats_client, js_context):
        from deepthought.eda.subscriber import Subscriber

        self._subscriber = Subscriber(nats_client, js_context)

    async def start(self, subject: str) -> None:
        await self._subscriber.subscribe(
            subject,
            self._handle,
            use_jetstream=True,
            durable="template",
        )

    @rate_limit(10, 1)  # 10 messages per second
    async def _handle(self, msg):
        start = time.perf_counter()
        await msg.ack()
        duration = time.perf_counter() - start
        INPUTS_TOTAL.labels(service="template_service").inc()
        INPUT_LATENCY_SECONDS.labels(service="template_service").observe(duration)
