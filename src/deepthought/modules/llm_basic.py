import json
import logging
from datetime import datetime, timezone
from typing import List, Optional

import torch
from nats.aio.client import Client as NATS
from nats.aio.msg import Msg
from nats.js.client import JetStreamContext
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..config import get_settings
from ..eda.events import EventSubjects, ResponseGeneratedPayload
from ..eda.publisher import Publisher
from ..eda.subscriber import Subscriber

logger = logging.getLogger(__name__)


class BasicLLM:
    """Simple LLM module using a small HuggingFace model."""

    def __init__(
        self,
        nats_client: Optional[NATS] = None,
        js_context: Optional[JetStreamContext] = None,
        model_name: Optional[str] = None,
    ) -> None:
        model_name = model_name or get_settings().model_path
        if nats_client is not None and js_context is not None:
            self._publisher: Optional[Publisher] = Publisher(nats_client, js_context)
            self._subscriber: Optional[Subscriber] = Subscriber(nats_client, js_context)
        else:
            self._publisher = None
            self._subscriber = None
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(model_name)
        logger.info("BasicLLM initialized with model %s", model_name)

    def _build_prompt(self, facts: List[str]) -> str:
        prompt = "\n".join(facts) + "\nResponse:" if facts else "Response:"
        return prompt

    async def _handle_memory_event(self, msg: Msg) -> None:
        input_id = "unknown"
        try:
            data = json.loads(msg.data.decode())
            input_id = data.get("input_id", "unknown")
            retrieved = data.get("retrieved_knowledge", {})
            if isinstance(retrieved, dict) and "retrieved_knowledge" in retrieved:
                knowledge = retrieved.get("retrieved_knowledge", {})
            elif isinstance(retrieved, dict):
                knowledge = retrieved
            else:
                logger.error("retrieved_knowledge is not a dict for input_id %s", input_id)
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
                return

            facts = knowledge.get("facts")
            if not isinstance(facts, list):
                logger.error("retrieved_knowledge missing facts list for input_id %s", input_id)
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
                return

            logger.info("BasicLLM received memory event ID %s", input_id)

            prompt = self._build_prompt([str(f) for f in facts])
            inputs = self._tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = self._model.generate(**inputs, max_length=inputs["input_ids"].shape[1] + 20)
            generated = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            if generated.startswith(prompt):
                response_text = generated[len(prompt) :].strip()  # noqa: E203
            else:
                response_text = generated.strip()

            payload = ResponseGeneratedPayload(
                final_response=response_text,
                input_id=input_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                confidence=0.5,
            )
            if self._publisher is not None:
                await self._publisher.publish(
                    EventSubjects.RESPONSE_GENERATED,
                    payload,
                    use_jetstream=True,
                    timeout=10.0,
                )
                logger.info("BasicLLM published RESPONSE_GENERATED for %s", input_id)
            else:
                logger.warning(
                    "Cannot publish RESPONSE_GENERATED for %s - publisher not initialized",
                    input_id,
                )
            await msg.ack()
        except Exception as e:
            logger.error("Error in BasicLLM handler: %s", e, exc_info=True)
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

    async def start_listening(self, durable_name: str = "llm_basic_listener") -> bool:
        if not self._subscriber:
            logger.error("Subscriber not initialized for BasicLLM.")
            return False
        try:
            await self._subscriber.subscribe(
                subject=EventSubjects.MEMORY_RETRIEVED,
                handler=self._handle_memory_event,
                use_jetstream=True,
                durable=durable_name,
            )
            logger.info("BasicLLM subscribed to %s", EventSubjects.MEMORY_RETRIEVED)
            return True
        except Exception as e:
            logger.error("BasicLLM failed to subscribe: %s", e, exc_info=True)
            return False

    async def stop_listening(self) -> None:
        if self._subscriber:
            await self._subscriber.unsubscribe_all()
            logger.info("BasicLLM stopped listening.")
        else:
            logger.warning("Cannot stop listening - no subscriber available.")
