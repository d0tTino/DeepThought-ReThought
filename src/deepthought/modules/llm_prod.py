import logging
import os
from typing import Optional

import nats
from nats.aio.client import Client as NATS
from nats.js.client import JetStreamContext
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..config import get_settings
from ..eda.events import EventSubjects
from ..eda.publisher import Publisher
from ..eda.subscriber import Subscriber
from .llm_base import BaseLLM

logger = logging.getLogger(__name__)


class ProductionLLM(BaseLLM):
    """LLM module using a base model merged with LoRA adapter weights."""

    def __init__(
        self,
        nats_client: NATS,
        js_context: JetStreamContext,
        model_name: Optional[str] = None,
        adapter_dir: str = "./results/lora-adapter",
    ) -> None:
        model_name = model_name or get_settings().model_path
        publisher = Publisher(nats_client, js_context)
        subscriber = Subscriber(nats_client, js_context)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        base_model = AutoModelForCausalLM.from_pretrained(model_name)
        if os.path.isdir(adapter_dir):
            logger.info("Loading LoRA adapter from %s", adapter_dir)
            model = PeftModel.from_pretrained(base_model, adapter_dir)
            model = model.merge_and_unload()
        else:
            logger.warning(
                "LoRA adapter directory %s not found. Using base model only.",
                adapter_dir,
            )
            model = base_model
        super().__init__(publisher, subscriber, tokenizer, model)
        logger.info("ProductionLLM initialized with model %s", model_name)

    async def start_listening(self, durable_name: str = "llm_prod_listener") -> bool:
        if self._subscriber is None:
            logger.error("Subscriber not initialized for ProductionLLM.")
            return False
        try:
            await self._subscriber.subscribe(
                subject=EventSubjects.MEMORY_RETRIEVED,
                handler=self._handle_memory_event,
                use_jetstream=True,
                durable=durable_name,
            )
            await self._subscriber.subscribe(
                subject="agent.reward",
                handler=self._handle_reward_event,
                use_jetstream=True,
                durable=f"{durable_name}_reward",
            )
            logger.info("ProductionLLM subscribed to %s", EventSubjects.MEMORY_RETRIEVED)
            return True
        except nats.errors.Error as e:
            logger.error("ProductionLLM failed to subscribe: %s", e, exc_info=True)
            return False
        except Exception as e:
            logger.error("ProductionLLM failed to subscribe: %s", e, exc_info=True)
            return False

    async def stop_listening(self) -> None:
        if self._subscriber:
            await self._subscriber.unsubscribe_all()
            logger.info("ProductionLLM stopped listening.")
        else:
            logger.warning("Cannot stop listening - no subscriber available.")
