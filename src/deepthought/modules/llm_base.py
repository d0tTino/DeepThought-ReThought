import json
import logging
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime, timezone
from typing import Deque, List, Optional

import nats
import torch
from contextlib import contextmanager
from nats.aio.msg import Msg

from ..config import get_settings
from ..eda.events import EventSubjects, ResponseGeneratedPayload
from ..eda.publisher import Publisher
from ..eda.subscriber import Subscriber


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


class BaseLLM(ABC):
    """Base class providing shared LLM functionality."""

    def __init__(
        self,
        publisher: Optional[Publisher],
        subscriber: Optional[Subscriber],
        tokenizer,
        model,
        reward_buffer_size: Optional[int] = None,
    ) -> None:
        self._publisher = publisher
        self._subscriber = subscriber
        self._tokenizer = tokenizer
        self._model = model
        buffer_size = reward_buffer_size or get_settings().reward.buffer_size
        self._recent_rewards: Deque[float] = deque(maxlen=buffer_size)

    @abstractmethod
    async def start_listening(self, durable_name: str = "llm_listener") -> bool:
        """Begin consuming events."""

    @abstractmethod
    async def stop_listening(self) -> None:
        """Stop consuming events."""

    def _build_prompt(self, facts: List[str]) -> str:
        """Assemble a prompt from retrieved facts and recent rewards."""
        reward_part = ""
        if self._recent_rewards:
            avg = sum(self._recent_rewards) / len(self._recent_rewards)
            reward_part = f"[avg_reward: {avg:.2f}]\n"
        base = "\n".join(facts) + "\nResponse:" if facts else "Response:"
        return reward_part + base

    async def _handle_memory_event(self, msg: Msg) -> None:
        """Common handler for MEMORY_RETRIEVED events."""
        input_id = "unknown"
        try:
            data = json.loads(msg.data.decode())
            if not isinstance(data, dict):
                raise ValueError("MemoryRetrieved payload must be a dict")
            input_id = data.get("input_id")
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

            prompt = self._build_prompt([str(f) for f in facts])
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
