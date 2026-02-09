"""HTTP-based LLM module for the demo."""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

import aiohttp
import nats
from nats.aio.client import Client as NATS
from nats.aio.msg import Msg
from nats.js.client import JetStreamContext

from ..eda.events import EventSubjects, ResponseCandidate, ResponseCandidatesPayload
from ..eda.publisher import Publisher
from ..eda.subscriber import Subscriber
from ..pipeline.dspy_pipeline import build_qa_pipeline

logger = logging.getLogger(__name__)


def _normalized_optional_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized if normalized else None


def _extract_memory_facts(retrieved: object) -> list[str]:
    if not isinstance(retrieved, dict):
        return []
    facts = retrieved.get("facts")
    if not isinstance(facts, list):
        return []
    return [str(fact) for fact in facts]


def _build_generation_prompt(
    *,
    user_input: str,
    facts: list[str],
    author_name: str | None = None,
    channel_context: str | None = None,
    recent_turn_summary: str | None = None,
) -> str:
    facts_block = "\n".join(f"- {fact}" for fact in facts) if facts else "- None"

    hints: list[str] = []
    if author_name:
        hints.append(f"- Author name: {author_name}")
    if channel_context:
        hints.append(f"- Channel context: {channel_context}")
    if recent_turn_summary:
        hints.append(f"- Recent turn summary: {recent_turn_summary}")
    hints_block = "\n".join(hints) if hints else "- None"

    return (
        "[SYSTEM PERSONA]\n"
        "You are DeepThought, a conversational assistant.\n"
        "Respond clearly, ground your answer in retrieved facts when relevant, and avoid inventing details.\n\n"
        "[RELEVANT FACTS]\n"
        f"{facts_block}\n\n"
        "[LATEST USER MESSAGE]\n"
        f"{user_input}\n\n"
        "[SOCIAL/PERCEPTION HINTS]\n"
        f"{hints_block}\n\n"
        "[TASK]\n"
        "Generate a helpful response to the user message."
    )


class RemoteLLM:
    """LLM module that calls a remote HTTP endpoint."""

    def __init__(self, nats_client: NATS, js_context: JetStreamContext, endpoint: Optional[str] = None) -> None:
        self._publisher = Publisher(nats_client, js_context)
        self._subscriber = Subscriber(nats_client, js_context)
        self._endpoint = endpoint or os.getenv("LLM_ENDPOINT", "http://localhost:8000/generate")
        if not self._endpoint:
            raise ValueError("LLM_ENDPOINT environment variable must be set or passed to RemoteLLM")
        self._session = aiohttp.ClientSession()

        use_dspy = os.getenv("USE_DSPY", "")
        self._use_dspy = use_dspy.lower() in {"1", "true", "yes", "on"}
        self._qa_pipeline = build_qa_pipeline() if self._use_dspy else None

        logger.info("RemoteLLM initialized with endpoint %s", self._endpoint)

    async def _generate(self, prompt: str) -> str:
        if self._use_dspy and self._qa_pipeline is not None:
            result = self._qa_pipeline(prompt)
            if not isinstance(result, str):
                raise ValueError("Invalid DSPy response")
            return result

        async with self._session.post(self._endpoint, json={"text": prompt}) as resp:
            resp.raise_for_status()
            data = await resp.json()
            text = data.get("text")
            if not isinstance(text, str):
                raise ValueError("Invalid generate response")
            return text

    async def _handle_memory_event(self, msg: Msg) -> None:
        input_id = "unknown"
        try:
            data = json.loads(msg.data.decode())
            if not isinstance(data, dict):
                raise ValueError("MemoryRetrieved payload must be a dict")
            input_id = data.get("input_id")
            author_id = data.get("author_id")
            if not isinstance(author_id, str):
                author_id = None
            user_id = data.get("user_id")
            if not isinstance(user_id, str):
                user_id = None
            if author_id is None:
                author_id = user_id
            channel_id = data.get("channel_id")
            if not isinstance(channel_id, str):
                channel_id = None
            user_input = _normalized_optional_text(data.get("user_input"))
            if user_input is None:
                raise ValueError("MemoryRetrieved payload missing required non-empty user_input")
            author_name = _normalized_optional_text(data.get("author_name"))
            channel_context = _normalized_optional_text(data.get("channel_context"))
            recent_turn_summary = _normalized_optional_text(data.get("recent_turn_summary"))
            facts = _extract_memory_facts(data.get("retrieved_knowledge", {}))
            prompt = _build_generation_prompt(
                user_input=user_input,
                facts=facts,
                author_name=author_name,
                channel_context=channel_context,
                recent_turn_summary=recent_turn_summary,
            )
            logger.info("RemoteLLM generating for %s", input_id)
            response = await self._generate(prompt)
            payload = ResponseCandidatesPayload(
                candidates=[
                    ResponseCandidate(
                        text=response,
                        confidence=0.9,
                        source="RemoteLLM",
                        safety_passed=True,
                    )
                ],
                input_id=input_id,
                user_id=author_id or user_id,
                author_id=author_id,
                channel_id=channel_id,
                timestamp=None,
            )
            await self._publisher.publish(
                EventSubjects.RESPONSE_CANDIDATES,
                payload,
                use_jetstream=True,
                timeout=10.0,
            )
            await msg.ack()
        except Exception as exc:  # pragma: no cover - runtime network or parse errors
            if hasattr(msg, "nak") and callable(msg.nak):
                try:
                    await msg.nak()
                except Exception:
                    logger.error("Failed to NAK message", exc_info=True)
            logger.exception("RemoteLLM failed: %s", exc)

    async def start_listening(self, durable_name: str = "remote_llm_listener") -> bool:
        if not self._subscriber:
            logger.error("Subscriber not initialized for RemoteLLM.")
            return False
        try:
            await self._subscriber.subscribe(
                subject=EventSubjects.MEMORY_RETRIEVED,
                handler=self._handle_memory_event,
                use_jetstream=True,
                durable=durable_name,
            )
            logger.info("RemoteLLM subscribed to %s", EventSubjects.MEMORY_RETRIEVED)
            return True
        except nats.errors.Error as e:
            logger.error("RemoteLLM failed to subscribe: %s", e, exc_info=True)
            return False
        except Exception as e:
            logger.error("RemoteLLM failed to subscribe: %s", e, exc_info=True)
            return False

    async def stop_listening(self) -> None:
        if self._subscriber:
            await self._subscriber.unsubscribe_all()
        if not self._session.closed:
            await self._session.close()
        logger.info("RemoteLLM stopped listening.")
