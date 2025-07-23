import logging
from typing import Awaitable, Callable, List, Tuple

from nats.aio.client import Client as NATS
from nats.aio.msg import Msg
from nats.js.client import JetStreamContext

from ..eda.publisher import Publisher
from ..eda.subscriber import Subscriber

logger = logging.getLogger(__name__)

MessageHandler = Callable[[Msg], Awaitable[None]]


class BaseService:
    """Base class providing subscription registration and lifecycle helpers.

    Subclasses call :meth:`add_subscription` to register NATS subjects and then
    use :meth:`start` and :meth:`stop` to manage them.
    """

    def __init__(self, nats_client: NATS, js_context: JetStreamContext) -> None:
        self._publisher = Publisher(nats_client, js_context)
        self._subscriber = Subscriber(nats_client, js_context)
        self._nc = nats_client
        self._subscriptions: List[Tuple[str, MessageHandler, str, str, bool]] = []

    def add_subscription(
        self,
        subject: str,
        handler: MessageHandler,
        *,
        durable: str = "",
        queue: str = "",
        use_jetstream: bool = False,
    ) -> None:
        """Register a subscription to be created on :meth:`start`."""
        self._subscriptions.append((subject, handler, durable, queue, use_jetstream))

    async def start(self) -> bool:
        """Create all registered subscriptions."""
        if self._subscriber is None:
            logger.error("Subscriber not initialized for %s.", self.__class__.__name__)
            return False
        success = True
        for subject, handler, durable, queue, use_js in self._subscriptions:
            res = await self._subscriber.subscribe(
                subject=subject,
                handler=handler,
                durable=durable,
                queue=queue,
                use_jetstream=use_js,
            )
            success = success and res
        return success

    async def stop(self) -> None:
        """Unsubscribe from all subjects and drain the NATS connection."""
        if self._subscriber:
            await self._subscriber.unsubscribe_all()
        if getattr(self, "_nc", None) and getattr(self._nc, "is_connected", False):
            await self._nc.drain()

    async def __aenter__(self) -> "BaseService":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.stop()
