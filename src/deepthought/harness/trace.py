import json
import logging

from nats.aio.client import Client as NATS
from nats.aio.msg import Msg
from nats.js.client import JetStreamContext

from ..eda.events import EventSubjects
from ..eda.subscriber import Subscriber

logger = logging.getLogger(__name__)


class TraceRecorder:
    """Record agent events to a JSONL file."""

    def __init__(
        self, nats_client: NATS, js_context: JetStreamContext, outfile: str
    ) -> None:
        if not outfile:
            raise ValueError("outfile path must be provided")
        self._subscriber = Subscriber(nats_client, js_context)
        self._outfile = outfile
        logger.info("TraceRecorder will write events to %s", outfile)

    async def _append(self, event: str, msg: Msg) -> None:
        try:
            data = json.loads(msg.data.decode())
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to decode %s payload: %s", event, e, exc_info=True)
            if hasattr(msg, "nak") and callable(msg.nak):
                try:
                    await msg.nak()
                except Exception:  # noqa: BLE001
                    logger.error("Failed to NAK message", exc_info=True)
            return

        record = {"event": event, "payload": data}
        try:
            with open(self._outfile, "a", encoding="utf-8") as f:
                json.dump(record, f)
                f.write("\n")
            await msg.ack()
            logger.debug("Recorded %s event", event)
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to record %s: %s", event, e, exc_info=True)
            if hasattr(msg, "nak") and callable(msg.nak):
                try:
                    await msg.nak()
                except Exception:  # noqa: BLE001
                    logger.error("Failed to NAK message after error", exc_info=True)

    async def _append_raw(self, event: str, msg: Msg) -> None:
        data = msg.data.decode()
        record = {"event": event, "payload": data}
        try:
            with open(self._outfile, "a", encoding="utf-8") as f:
                json.dump(record, f)
                f.write("\n")
            await msg.ack()
            logger.debug("Recorded %s event", event)
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to record %s: %s", event, e, exc_info=True)
            if hasattr(msg, "nak") and callable(msg.nak):
                try:
                    await msg.nak()
                except Exception:  # noqa: BLE001
                    logger.error("Failed to NAK message after error", exc_info=True)

    async def _handle_input(self, msg: Msg) -> None:
        await self._append("INPUT_RECEIVED", msg)

    async def _handle_response(self, msg: Msg) -> None:
        await self._append("RESPONSE_GENERATED", msg)

    async def _handle_chat_raw(self, msg: Msg) -> None:
        await self._append_raw("CHAT_RAW", msg)

    async def start(self, durable_name: str = "trace_recorder") -> bool:
        try:
            await self._subscriber.subscribe(
                subject=EventSubjects.INPUT_RECEIVED,
                handler=self._handle_input,
                use_jetstream=True,
                durable=f"{durable_name}_input",
            )
            await self._subscriber.subscribe(
                subject=EventSubjects.RESPONSE_GENERATED,
                handler=self._handle_response,
                use_jetstream=True,
                durable=f"{durable_name}_response",
            )
            await self._subscriber.subscribe(
                subject=EventSubjects.CHAT_RAW,
                handler=self._handle_chat_raw,
                use_jetstream=True,
                durable=f"{durable_name}_chat_raw",
            )
            logger.info("TraceRecorder subscribed to events")
            return True
        except Exception as e:  # noqa: BLE001
            logger.error("TraceRecorder failed to subscribe: %s", e, exc_info=True)
            return False

    async def stop(self) -> None:
        if self._subscriber:
            await self._subscriber.unsubscribe_all()
            logger.info("TraceRecorder stopped listening")
