import logging
from typing import Optional

import nats
from nats.aio.client import Client as NATS
from nats.js.client import JetStreamContext
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..config import get_settings
from ..eda.events import EventSubjects
from ..eda.publisher import Publisher
from ..eda.subscriber import Subscriber
from .llm_base import BaseLLM

logger = logging.getLogger(__name__)


class BasicLLM(BaseLLM):
    """Simple LLM module using a small HuggingFace model."""

    def __init__(
        self,
        nats_client: Optional[NATS] = None,
        js_context: Optional[JetStreamContext] = None,
        model_name: Optional[str] = None,
    ) -> None:
        model_name = model_name or get_settings().model_path
        if nats_client is not None and js_context is not None:
            publisher = Publisher(nats_client, js_context)
            subscriber = Subscriber(nats_client, js_context)
        else:
            publisher = None
            subscriber = None
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        super().__init__(publisher, subscriber, tokenizer, model)
        logger.info("BasicLLM initialized with model %s", model_name)

    async def start_listening(self, durable_name: str = "llm_basic_listener") -> bool:
        if self._subscriber is None:
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
        except nats.errors.Error as e:
            logger.error("BasicLLM failed to subscribe: %s", e, exc_info=True)
            return False
        except Exception as e:
            logger.error("BasicLLM failed to subscribe: %s", e, exc_info=True)
            return False

    async def stop_listening(self) -> None:
        if self._subscriber:
            await self._subscriber.unsubscribe_all()
            logger.info("BasicLLM stopped listening.")
        else:
            logger.warning("Cannot stop listening - no subscriber available.")
