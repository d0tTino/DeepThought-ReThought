"""Shared bus interfaces for DeepThought services."""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Protocol

from .eda.publisher import Publisher as _Publisher
from .eda.subscriber import Subscriber as _Subscriber

MessageHandler = Callable[[Any], Awaitable[None]]


class Publisher(Protocol):
    """Protocol describing the publish interface used by services."""

    async def publish(self, subject: str, payload: Any) -> None:
        ...


class Subscriber(Protocol):
    """Protocol describing the subscribe interface used by services."""

    async def subscribe(
        self,
        subject: str,
        handler: MessageHandler,
        queue: str = "",
        use_jetstream: bool = False,
        durable: str = "",
    ) -> None:
        ...


__all__ = ["Publisher", "Subscriber", "MessageHandler", "get_publisher", "get_subscriber"]


def get_publisher(publisher: _Publisher) -> Publisher:
    """Return a Publisher compliant instance from the EDA layer."""

    return publisher


def get_subscriber(subscriber: _Subscriber) -> Subscriber:
    """Return a Subscriber compliant instance from the EDA layer."""

    return subscriber
