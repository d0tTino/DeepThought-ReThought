# File: src/deepthought/eda/publisher.py
import asyncio
import logging
import ssl

# Imported lazily in ``connect`` to avoid mandatory dependencies
from typing import Any, Dict, Optional, Union

import nats
from nats.aio.client import Client as NATS
from nats.js.client import JetStreamContext

from .contracts import EventEnvelope, validate_cross_service_envelope

logger = logging.getLogger(__name__)
NATS_TIMEOUT_ERROR = getattr(nats.errors, "TimeoutError", TimeoutError)


class Publisher:
    """A publisher using a shared NATS client and JetStream context."""

    def __init__(self, nats_client: NATS, js_context: JetStreamContext):
        """Initialize Publisher with existing client and context."""
        if not nats_client or (hasattr(nats_client, "is_connected") and not nats_client.is_connected):
            raise ValueError("NATS client must be connected.")
        if not js_context:
            raise ValueError("JetStream context must be provided.")
        self._nc = nats_client
        self._js = js_context
        logger.debug("Publisher initialized with shared client and JS context.")

    async def publish(
        self,
        subject: str,
        payload: Union[str, Dict, Any],
        use_jetstream: bool = True,
        timeout: float = 10.0,
        retries: int = 3,
    ) -> Optional[Dict]:
        """Publish message, retrying for at-least-once semantics."""
        if retries < 1:
            raise ValueError("retries must be at least 1")

        # Convert payload
        if isinstance(payload, bytes):
            data = payload
        elif isinstance(payload, str):
            data = payload.encode()
        elif hasattr(payload, "to_json"):
            data = payload.to_json().encode()
        elif isinstance(payload, (dict, list)):
            import json

            data = json.dumps(payload).encode()
        else:
            data = str(payload).encode()

        payload_summary = str(payload)
        if len(payload_summary) > 100:
            payload_summary = payload_summary[:97] + "..."

        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                if use_jetstream:
                    ack = await self._js.publish(subject, data, timeout=timeout)
                    logger.debug("Published to '%s' via JetStream: seq=%s", subject, ack.seq)
                    return {"seq": ack.seq, "stream": ack.stream}
                # Use regular NATS publish
                await self._nc.publish(subject, data)
                logger.debug("Published basic NATS message to '%s'", subject)
                return None
            except NATS_TIMEOUT_ERROR as err:
                last_error = err
                logger.warning("Publish timeout for '%s' with payload %s: %s", subject, payload_summary, err)
            except Exception as err:
                last_error = err
                logger.warning(
                    "Failed to publish to '%s' with payload %s: %s",
                    subject,
                    payload_summary,
                    err,
                )
            if attempt < retries:
                await asyncio.sleep(min(0.1 * attempt, 1.0))

        assert last_error is not None
        logger.error("Failed to publish to '%s' after %s attempts", subject, retries)
        raise last_error


async def connect(
    nats_url: str | None = None,
    *,
    tls_cert: str | None = None,
    tls_key: str | None = None,
    tls_ca: str | None = None,
    user: str | None = None,
    password: str | None = None,
    name: str = "dtrt_publisher",
) -> "Publisher":
    """Create a :class:`Publisher` connected to ``nats_url`` with optional TLS."""

    from ..config import get_settings

    settings = get_settings()
    nats_url = nats_url or settings.nats_url
    tls_cert = tls_cert or settings.nats_tls_cert
    tls_key = tls_key or settings.nats_tls_key
    tls_ca = tls_ca or settings.nats_tls_ca
    user = user or settings.nats_username
    password = password or settings.nats_password

    ssl_ctx = None
    if tls_cert and tls_key:
        ssl_ctx = ssl.create_default_context(purpose=ssl.Purpose.SERVER_AUTH)
        if tls_ca:
            ssl_ctx.load_verify_locations(tls_ca)
        ssl_ctx.load_cert_chain(tls_cert, tls_key)

    nc = NATS()
    await nc.connect(
        servers=[nats_url],
        tls=ssl_ctx,
        user=user,
        password=password,
        name=name,
    )
    js = nc.jetstream()
    return Publisher(nc, js)


async def publish_enveloped(
    publisher: "Publisher",
    *,
    subject: str,
    payload: Dict[str, Any],
    producer: str,
    trace_id: str | None = None,
    causation_id: str | None = None,
    use_jetstream: bool = True,
    timeout: float = 10.0,
    retries: int = 3,
) -> Optional[Dict]:
    """Build, validate, and publish a canonical EventEnvelope."""

    envelope = EventEnvelope.build(
        subject=subject,
        payload=payload,
        producer=producer,
        trace_id=trace_id,
        causation_id=causation_id,
    )
    validate_cross_service_envelope(subject, envelope.__dict__)
    try:
        return await publisher.publish(
            subject,
            envelope.__dict__,
            use_jetstream=use_jetstream,
            timeout=timeout,
            retries=retries,
        )
    except TypeError as exc:
        if "unexpected keyword argument 'retries'" not in str(exc):
            raise
        return await publisher.publish(
            subject,
            envelope.__dict__,
            use_jetstream=use_jetstream,
            timeout=timeout,
        )
