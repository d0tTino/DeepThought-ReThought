import json
import logging
from abc import ABC, abstractmethod
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Deque, Iterable, List, Optional

import nats
import torch
from nats.aio.msg import Msg

from ..config import get_settings
from ..eda.events import EventSubjects, ResponseGeneratedPayload
from ..eda.publisher import Publisher
from ..eda.subscriber import Subscriber
from ..services.response_filter import build_response_filter

logger = logging.getLogger(__name__)


@contextmanager
def _safe_no_grad():
    """Fallback context manager if torch.no_grad fails."""
    cm = getattr(torch, "no_grad", None)
    if cm is None:
        yield
        return
    try:
        with cm():
            yield
    except Exception:  # pragma: no cover - fallback
        yield


def build_prompt(parts: Iterable[str], persona_desc: str | None = None, reward_context: str = "") -> str:
    """Create a prompt from context lines, persona description, and reward metadata."""
    base = "\n".join(parts)
    if base:
        base = f"{base}\nResponse:"
    else:
        base = "Response:"
    prompt = f"{reward_context}{base}"
    if persona_desc:
        prompt = persona_desc.strip() + "\n" + prompt
    return prompt


class BaseLLM(ABC):
    """Base class providing shared LLM functionality."""

    def __init__(
        self,
        publisher: Optional[Publisher],
        subscriber: Optional[Subscriber],
        tokenizer,
        model,
        reward_buffer_size: Optional[int] = None,
        persona_manager=None,
    ) -> None:
        self._publisher = publisher
        self._subscriber = subscriber
        self._tokenizer = tokenizer
        self._model = model
        settings = get_settings()
        buffer_size = reward_buffer_size or settings.reward.buffer_size
        self._recent_rewards: Deque[float] = deque(maxlen=buffer_size)
        self._persona_manager = persona_manager
        self._persona_descriptions = settings.persona_descriptions
        self._response_filter_enabled = settings.response_filter_enabled
        self._response_filter_fallback = settings.response_filter_fallback_message
        self._response_filter = build_response_filter(
            settings.response_filter_denylist,
            settings.response_filter_classifier,
        )

    @abstractmethod
    async def start_listening(self, durable_name: str = "llm_listener") -> bool:
        """Begin consuming events."""

    @abstractmethod
    async def stop_listening(self) -> None:
        """Stop consuming events."""

    def _build_prompt(self, facts: List[str], persona_desc: str | None = None) -> str:
        """Assemble a prompt from persona, retrieved facts and rewards."""
        reward_part = ""
        if self._recent_rewards:
            avg = sum(self._recent_rewards) / len(self._recent_rewards)
            reward_part = f"[avg_reward: {avg:.2f}]\n"
        if facts:
            memory_lines = "\n".join(f"- {fact}" for fact in facts)
            base = f"MEMORY_RETRIEVED:\n{memory_lines}\nResponse:"
        else:
            base = "Response:"
        prompt = reward_part + base
        if persona_desc:
            prompt = persona_desc.strip() + "\n" + prompt
        return prompt

    async def _handle_memory_event(self, msg: Msg) -> None:
        """Common handler for MEMORY_RETRIEVED events."""
        input_id = "unknown"
        try:
            data = json.loads(msg.data.decode())
            if not isinstance(data, dict):
                raise ValueError("MemoryRetrieved payload must be a dict")
            input_id = data.get("input_id")
            user_id = data.get("user_id")
            if not isinstance(user_id, str):
                user_id = None
            target_id = data.get("target_id") or (msg.headers.get("target_id") if msg.headers else None)
            if not isinstance(target_id, str):
                target_id = None
            knowledge = data.get("retrieved_knowledge")
            if not isinstance(input_id, str) or not isinstance(knowledge, dict):
                raise ValueError("Invalid memory payload fields")

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

            logger.info("%s received memory event ID %s", self.__class__.__name__, input_id)

            persona_desc = ""
            if self._persona_manager is not None:
                try:
                    persona_id = user_id if user_id is not None else input_id
                    persona_desc = await self._persona_manager.get_description(persona_id, target_id)
                except Exception:
                    logger.error("Persona selection failed", exc_info=True)

            prompt = self._build_prompt([str(f) for f in facts], persona_desc)
            inputs = self._tokenizer(prompt, return_tensors="pt")
            with _safe_no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_length=inputs["input_ids"].shape[1] + 20,
                )
            generated = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            if generated.startswith(prompt):
                response_text = generated[len(prompt) :].strip()  # noqa: E203
            else:
                response_text = generated.strip()

            if self._response_filter_enabled and not self._response_filter.is_safe(response_text):
                logger.warning(
                    "Response filtered for input_id %s; using fallback response",
                    input_id,
                )
                response_text = self._response_filter_fallback

            payload = ResponseGeneratedPayload(
                final_response=response_text,
                input_id=input_id,
                user_id=user_id,
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
                logger.info("%s published RESPONSE_GENERATED for %s", self.__class__.__name__, input_id)
            else:
                logger.warning(
                    "Cannot publish RESPONSE_GENERATED for %s - publisher not initialized",
                    input_id,
                )
            await msg.ack()
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("Invalid MemoryRetrieved payload: %s", e, exc_info=True)
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

        except Exception as e:
            logger.error("Error in %s handler: %s", self.__class__.__name__, e, exc_info=True)
            if hasattr(msg, "nak") and callable(msg.nak):
                try:
                    await msg.nak()
                except nats.errors.Error:
                    logger.error("Failed to NAK message", exc_info=True)
            elif hasattr(msg, "ack") and callable(msg.ack):
                try:
                    await msg.ack()
                except nats.errors.Error:
                    logger.error("Failed to ack message after error", exc_info=True)

    async def _handle_reward_event(self, msg: Msg) -> None:
        """Store rewards published on ``agent.reward``."""
        try:
            data = json.loads(msg.data.decode())
            if not isinstance(data, dict) or "reward" not in data:
                raise ValueError("payload must contain reward field")
            reward = float(data["reward"])
            self._recent_rewards.append(reward)
        except Exception as exc:  # pragma: no cover - invalid payload
            logger.error("Invalid agent.reward payload: %s", exc)
        finally:
            if hasattr(msg, "ack") and callable(msg.ack):
                try:
                    await msg.ack()
                except Exception:  # pragma: no cover - ack issues
                    logger.error("Failed to ack reward message", exc_info=True)
