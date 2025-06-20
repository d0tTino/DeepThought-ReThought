import json
import logging
import os
from datetime import datetime, timezone
from typing import List, Optional

import torch
from nats.aio.client import Client as NATS
from nats.aio.msg import Msg
from nats.js.client import JetStreamContext
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..config import get_settings
from ..eda.events import EventSubjects, ResponseGeneratedPayload
from ..eda.publisher import Publisher
from ..eda.subscriber import Subscriber

logger = logging.getLogger(__name__)


class ProductionLLM:
    """LLM module using a base model merged with LoRA adapter weights."""

    def __init__(
        self,
        nats_client: NATS,
        js_context: JetStreamContext,
        model_name: Optional[str] = None,
        adapter_dir: str = "./results/lora-adapter",
    ) -> None:
        model_name = model_name or get_settings().model_path
        self._publisher = Publisher(nats_client, js_context)
        self._subscriber = Subscriber(nats_client, js_context)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        base_model = AutoModelForCausalLM.from_pretrained(model_name)
        if os.path.isdir(adapter_dir):
            logger.info("Loading LoRA adapter from %s", adapter_dir)
            model = PeftModel.from_pretrained(base_model, adapter_dir)
            self._model = model.merge_and_unload()
        else:
            logger.warning(
                "LoRA adapter directory %s not found. Using base model only.",
                adapter_dir,
            )
            self._model = base_model
        logger.info("ProductionLLM initialized with model %s", model_name)

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

            logger.info("ProductionLLM received memory event ID %s", input_id)

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
            await self._publisher.publish(EventSubjects.RESPONSE_GENERATED, payload, use_jetstream=True, timeout=10.0)
            logger.info("ProductionLLM published RESPONSE_GENERATED for %s", input_id)
            await msg.ack()
        except Exception as e:  # pragma: no cover - runtime errors are logged
            logger.error("Error in ProductionLLM handler: %s", e, exc_info=True)
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

    async def start_listening(self, durable_name: str = "llm_prod_listener") -> bool:
        if not self._subscriber:
            logger.error("Subscriber not initialized for ProductionLLM.")
            return False
        try:
            await self._subscriber.subscribe(
                subject=EventSubjects.MEMORY_RETRIEVED,
                handler=self._handle_memory_event,
                use_jetstream=True,
                durable=durable_name,
            )
            logger.info("ProductionLLM subscribed to %s", EventSubjects.MEMORY_RETRIEVED)
            return True
        except Exception as e:
            logger.error("ProductionLLM failed to subscribe: %s", e, exc_info=True)
            return False

    async def stop_listening(self) -> None:
        if self._subscriber:
            await self._subscriber.unsubscribe_all()
            logger.info("ProductionLLM stopped listening.")
        else:
            logger.warning("Cannot stop listening - no subscriber available.")
