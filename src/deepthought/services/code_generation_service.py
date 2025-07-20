import json
import logging
from datetime import datetime, timezone
from string import Template
from typing import Any, Dict

import nats
from nats.aio.client import Client as NATS
from nats.aio.msg import Msg
from nats.js.client import JetStreamContext

from ..eda.events import (
    CodeGeneratedPayload,
    EventSubjects,
)
from ..eda.publisher import Publisher
from ..eda.subscriber import Subscriber

logger = logging.getLogger(__name__)


class CodeGenerationService:
    """Simple template-based code generation service."""

    def __init__(self, nats_client: NATS, js_context: JetStreamContext) -> None:
        self._publisher = Publisher(nats_client, js_context)
        self._subscriber = Subscriber(nats_client, js_context)

    async def _handle_template_request(self, msg: Msg) -> None:
        request_id = "unknown"
        try:
            data = json.loads(msg.data.decode())
            if not isinstance(data, dict):
                raise ValueError("CodeTemplate payload must be a dict")
            request_id = data.get("input_id")
            template_text = data.get("template")
            variables = data.get("variables", {})
            if not isinstance(template_text, str) or not isinstance(variables, dict):
                raise ValueError("Invalid template payload fields")
            logger.info("CodeGenerationService received request ID %s", request_id)

            code = Template(template_text).safe_substitute(**variables)
            local_ns: Dict[str, Any] = {}
            exec(code, {}, local_ns)
            result = local_ns.get("result")

            payload = CodeGeneratedPayload(
                code=code,
                result=str(result),
                input_id=request_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            await self._publisher.publish(
                EventSubjects.CODE_GENERATED,
                payload,
                use_jetstream=True,
                timeout=10.0,
            )
            logger.info("CodeGenerationService published generated code event ID %s", request_id)
            await msg.ack()
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("Invalid CodeTemplate payload: %s", e, exc_info=True)
            if hasattr(msg, "nak") and callable(msg.nak):
                try:
                    await msg.nak()
                except Exception:
                    logger.error("Failed to NAK message", exc_info=True)
            elif hasattr(msg, "ack") and callable(msg.ack):
                try:
                    await msg.ack()
                except Exception:
                    logger.error("Failed to ack message after error", exc_info=True)
        except Exception as e:  # pragma: no cover - unexpected failures
            logger.error("Error in CodeGenerationService handler: %s", e, exc_info=True)
            if hasattr(msg, "nak") and callable(msg.nak):
                try:
                    await msg.nak()
                except Exception:
                    logger.error("Failed to NAK message", exc_info=True)
            elif hasattr(msg, "ack") and callable(msg.ack):
                try:
                    await msg.ack()
                except Exception:
                    logger.error("Failed to ack message after error", exc_info=True)

    async def start(self, durable_name: str = "codegen_listener") -> bool:
        if self._subscriber is None:
            logger.error("Subscriber not initialized for CodeGenerationService.")
            return False
        try:
            await self._subscriber.subscribe(
                subject=EventSubjects.CODE_TEMPLATE_REQUEST,
                handler=self._handle_template_request,
                use_jetstream=True,
                durable=durable_name,
            )
            logger.info("CodeGenerationService subscribed to %s", EventSubjects.CODE_TEMPLATE_REQUEST)
            return True
        except nats.errors.Error as e:
            logger.error("CodeGenerationService failed to subscribe: %s", e, exc_info=True)
            return False
        except Exception as e:
            logger.error("CodeGenerationService failed to subscribe: %s", e, exc_info=True)
            return False

    async def stop(self) -> None:
        if self._subscriber:
            await self._subscriber.unsubscribe_all()
            logger.info("CodeGenerationService stopped listening.")
        else:
            logger.warning("Cannot stop listening - no subscriber available.")
