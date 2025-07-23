import asyncio
import logging
from typing import Awaitable, Callable, List, Optional, Tuple

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

    def __init__(
        self,
        nats_client: Optional[NATS] = None,
        js_context: Optional[JetStreamContext] = None,
        *,
        nats_url: str | None = None,
        connect_retries: int = 1,
        connect_timeout: float = 2.0,
    ) -> None:
        self._nats_url = nats_url
        self._connect_retries = max(connect_retries, 1)
        self._connect_timeout = connect_timeout
        self._nc = nats_client or NATS()
        self._js = js_context
        self._publisher: Optional[Publisher] = None
        self._subscriber: Optional[Subscriber] = None
        if self._nc and getattr(self._nc, "is_connected", False) and self._js is None:
            self._js = self._nc.jetstream()
        if self._nc and getattr(self._nc, "is_connected", False) and self._js:
            self._publisher = Publisher(self._nc, self._js)
            self._subscriber = Subscriber(self._nc, self._js)
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

    async def _ensure_connected(self) -> bool:
        if getattr(self._nc, "is_connected", False):
            return True
        if not hasattr(self._nc, "connect") or not self._nats_url:
            logger.error("NATS connection unavailable for %s", self.__class__.__name__)
            return False
        for attempt in range(1, self._connect_retries + 1):
            try:
                await self._nc.connect(servers=[self._nats_url], connect_timeout=self._connect_timeout)
                self._js = self._nc.jetstream()
                self._publisher = Publisher(self._nc, self._js)
                self._subscriber = Subscriber(self._nc, self._js)
                return True
            except Exception as exc:
                if attempt >= self._connect_retries:
                    logger.error(
                        "Failed to connect to NATS after %d attempts: %s", self._connect_retries, exc
                    )
                    return False
                logger.warning("NATS connect attempt %d failed: %s", attempt, exc)
                await asyncio.sleep(0.5)
        return False

    async def start(self) -> bool:
        """Create all registered subscriptions."""
        if not await self._ensure_connected():
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
