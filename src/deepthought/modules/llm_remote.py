"""LLM responder module with pluggable backends."""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from typing import Protocol

import aiohttp
import nats
from nats.aio.client import Client as NATS
from nats.aio.msg import Msg
from nats.js.client import JetStreamContext

from ..config import get_settings
from ..eda.contracts import EventEnvelope, decode_payload_or_envelope
from ..eda.events import ContextAssembledPayload, EventSubjects, ResponseCandidate, ResponseCandidatesPayload
from ..eda.publisher import Publisher
from ..eda.subscriber import Subscriber
from ..pipeline.dspy_pipeline import build_qa_pipeline

logger = logging.getLogger(__name__)


class ResponderBackend(Protocol):
    """Backend interface for generating response candidates."""

    async def generate_candidates(self, prompt: str, num_candidates: int) -> list[dict[str, object]]:
        """Return backend candidates as generic dictionaries."""

    async def close(self) -> None:
        """Release backend resources."""


@dataclass
class HTTPResponderBackend:
    endpoint: str
    session: aiohttp.ClientSession

    async def generate_candidates(self, prompt: str, num_candidates: int) -> list[dict[str, object]]:
        async with self.session.post(self.endpoint, json={"text": prompt, "n": num_candidates}) as resp:
            resp.raise_for_status()
            data = await resp.json()
            raw_candidates = data.get("candidates") if isinstance(data, dict) else None
            if isinstance(raw_candidates, list) and raw_candidates:
                return [item for item in raw_candidates if isinstance(item, dict)]

            text = data.get("text") if isinstance(data, dict) else None
            if not isinstance(text, str):
                raise ValueError("Invalid generate response")
            return [{"text": text, "source": "remote_http"}]

    async def close(self) -> None:
        return None


class LocalQuantizedResponderBackend:
    """Local HuggingFace backend with optional quantization."""

    def __init__(self, model_path: str, quantization_bits: int | None = None, max_new_tokens: int = 192) -> None:
        self._model_path = model_path
        self._quantization_bits = quantization_bits
        self._max_new_tokens = max_new_tokens
        self._generator = None

    def _build_pipeline(self):
        if self._generator is not None:
            return self._generator

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
        except Exception as exc:  # pragma: no cover - dependency optional in many environments
            raise RuntimeError("transformers package is required for local_quantized backend") from exc

        model_kwargs: dict[str, object] = {"device_map": "auto"}
        if self._quantization_bits in {4, 8}:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=self._quantization_bits == 4,
                load_in_8bit=self._quantization_bits == 8,
            )

        tokenizer = AutoTokenizer.from_pretrained(self._model_path)
        model = AutoModelForCausalLM.from_pretrained(self._model_path, **model_kwargs)
        self._generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
        return self._generator

    async def generate_candidates(self, prompt: str, num_candidates: int) -> list[dict[str, object]]:
        generator = self._build_pipeline()
        outputs = generator(
            prompt,
            num_return_sequences=num_candidates,
            max_new_tokens=self._max_new_tokens,
            do_sample=True,
            return_full_text=False,
        )
        if not isinstance(outputs, list) or not outputs:
            raise ValueError("Local backend did not produce outputs")

        candidates: list[dict[str, object]] = []
        for item in outputs:
            if not isinstance(item, dict):
                continue
            text = item.get("generated_text")
            if isinstance(text, str) and text.strip():
                candidates.append({"text": text, "source": "local_quantized"})
        if not candidates:
            raise ValueError("Local backend produced no valid text candidates")
        return candidates

    async def close(self) -> None:
        self._generator = None


def _normalized_optional_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized if normalized else None


def _extract_memory_facts(data: dict[str, object]) -> list[str]:
    facts = data.get("retrieved_facts")
    if isinstance(facts, list):
        return [str(fact) for fact in facts]
    retrieved = data.get("retrieved_knowledge")
    if not isinstance(retrieved, dict):
        return []
    fallback_facts = retrieved.get("facts")
    if not isinstance(fallback_facts, list):
        return []
    return [str(fact) for fact in fallback_facts]


def _build_generation_prompt(
    *,
    user_input: str,
    facts: list[str],
    author_name: str | None = None,
    channel_context: str | None = None,
    recent_turn_summary: str | None = None,
    multimodal_interpretations: dict[str, object] | None = None,
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

    multimodal = multimodal_interpretations if isinstance(multimodal_interpretations, dict) else {}
    modality_notes = multimodal.get("notes")
    notes = modality_notes if isinstance(modality_notes, list) else []
    note_lines: list[str] = []
    for raw_note in notes:
        if not isinstance(raw_note, dict):
            continue
        modality = str(raw_note.get("modality", "unknown"))
        what = str(raw_note.get("what", "unknown signal"))
        where = str(raw_note.get("where", "unknown location"))
        who = str(raw_note.get("who", "unknown actor"))
        confidence = raw_note.get("confidence")
        conf_txt = f"{float(confidence):.2f}" if isinstance(confidence, (int, float)) else "n/a"
        note_lines.append(f"- [{modality}] what={what}; where={where}; who={who}; confidence={conf_txt}")
    multimodal_block = "\n".join(note_lines) if note_lines else "- None"

    multimodal_confidence = multimodal.get("confidence") if isinstance(multimodal.get("confidence"), dict) else {}
    aggregate = multimodal_confidence.get("aggregate")
    low_confidence = bool(multimodal_confidence.get("low_confidence"))
    uncertainty_line = (
        f"Multimodal confidence={float(aggregate):.2f}; low_confidence={low_confidence}."
        if isinstance(aggregate, (int, float))
        else f"Multimodal confidence unknown; low_confidence={low_confidence}."
    )

    fallback = multimodal.get("fallback") if isinstance(multimodal.get("fallback"), dict) else {}
    clarify = bool(fallback.get("ask_clarifying_question"))

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
        "[MULTIMODAL INTERPRETATIONS]\n"
        f"{multimodal_block}\n\n"
        "[UNCERTAINTY CUES]\n"
        f"- {uncertainty_line}\n\n"
        "[TASK]\n"
        + (
            "Ask a focused clarifying question before making claims about image/audio details."
            if clarify or low_confidence
            else "Generate a helpful response to the user message."
        )
    )


def _build_clarifying_question(multimodal_interpretations: dict[str, object] | None) -> str:
    multimodal = multimodal_interpretations if isinstance(multimodal_interpretations, dict) else {}
    notes = multimodal.get("notes")
    if isinstance(notes, list):
        modalities = [str(item.get("modality")) for item in notes if isinstance(item, dict) and item.get("modality")]
        if modalities:
            joined = ", ".join(sorted(set(modalities)))
            return f"I might be missing details from the {joined} signal. Could you clarify what you want me to focus on?"
    return "I may be missing key details from the attachment. Could you clarify what is most important for me to analyze?"


class RemoteLLM:
    """LLM module that generates response candidates via configured backend."""

    def __init__(self, nats_client: NATS, js_context: JetStreamContext, endpoint: str | None = None) -> None:
        self._publisher = Publisher(nats_client, js_context)
        self._subscriber = Subscriber(nats_client, js_context)
        self._session = aiohttp.ClientSession()

        settings = get_settings()
        self._num_candidates = max(1, int(os.getenv("LLM_NUM_CANDIDATES", "3")))
        self._source_name = os.getenv("LLM_SOURCE_NAME", "remote_llm")

        use_dspy = os.getenv("USE_DSPY", "")
        self._use_dspy = use_dspy.lower() in {"1", "true", "yes", "on"}
        self._qa_pipeline = build_qa_pipeline() if self._use_dspy else None

        env_backend = os.getenv("LLM_BACKEND") or os.getenv("DT_LLM_BACKEND")
        requested_backend = (env_backend or settings.llm_backend or "remote_http").strip().lower()

        endpoint_value = endpoint if endpoint is not None else os.getenv("LLM_ENDPOINT", settings.llm_remote_endpoint)
        if requested_backend == "remote_http":
            if not endpoint_value:
                raise ValueError("LLM endpoint must be set for remote_http backend")
            self._backend: ResponderBackend = HTTPResponderBackend(endpoint_value, self._session)
        elif requested_backend == "local_quantized":
            model_path = settings.llm_model_path or settings.model_path
            if not model_path:
                raise ValueError("A model path is required for local_quantized backend")
            self._backend = LocalQuantizedResponderBackend(
                model_path=model_path,
                quantization_bits=settings.llm_quantization_bits,
                max_new_tokens=settings.llm_local_max_new_tokens,
            )
        else:
            raise ValueError(f"Unsupported LLM backend: {requested_backend}")

        self._backend_name = requested_backend
        logger.info("RemoteLLM initialized with backend=%s", self._backend_name)

    def _calibrate_confidence(self, candidate: dict[str, object], text: str) -> tuple[float, dict[str, float]]:
        token_score = 0.0
        raw_logprob = candidate.get("avg_logprob")
        if isinstance(raw_logprob, (float, int)):
            token_score = 1.0 / (1.0 + math.exp(-float(raw_logprob)))

        model_score = 0.0
        raw_model_score = candidate.get("score")
        if isinstance(raw_model_score, (float, int)):
            model_score = max(0.0, min(1.0, float(raw_model_score)))

        length = len(text.split())
        length_score = min(1.0, max(0.0, length / 32.0))
        confidence = (0.5 * token_score) + (0.35 * model_score) + (0.15 * length_score)
        components = {
            "token_score": round(token_score, 4),
            "model_score": round(model_score, 4),
            "length_score": round(length_score, 4),
        }
        return round(max(0.0, min(1.0, confidence)), 4), components

    def _evaluate_safety(self, text: str) -> tuple[bool, dict[str, object]]:
        lowered = text.lower()
        blocked_terms = ["kill", "bomb", "dox", "self-harm"]
        matched = [term for term in blocked_terms if term in lowered]
        safe = not matched
        return safe, {
            "rule": "keyword_v1",
            "matched_terms": matched,
            "severity": "high" if matched else "none",
            "safety_passed": safe,
        }

    def _build_grounded_metadata(
        self,
        *,
        source: str,
        confidence: float,
        confidence_components: dict[str, float],
        safety_passed: bool,
        safety_metadata: dict[str, object],
        prompt: str | None = None,
        response_mode: str = "generative",
    ) -> dict[str, object]:
        calibration = {
            "slope": 1.0,
            "bias": 0.0,
            "version": "remote_llm_v1",
        }
        return {
            "source": source,
            "backend": self._backend_name,
            "response_mode": response_mode,
            "is_primary_voice": True,
            "confidence": confidence,
            "safety_passed": safety_passed,
            "calibration": calibration,
            "calibration_metadata": {
                "calibration": calibration,
                "confidence_components": confidence_components,
                "prompt_chars": len(prompt or ""),
                "candidate_count": self._num_candidates,
            },
            "grounding": {
                "contract": "canonical_conversational_responder_v1",
                "prompt_grounded": bool(prompt),
                "safety_rule": safety_metadata.get("rule"),
            },
        }

    async def _generate_candidates(self, prompt: str) -> list[ResponseCandidate]:
        if self._use_dspy and self._qa_pipeline is not None:
            result = self._qa_pipeline(prompt)
            if not isinstance(result, str):
                raise ValueError("Invalid DSPy response")
            safe, safety_metadata = self._evaluate_safety(result)
            confidence, components = self._calibrate_confidence({}, result)
            return [
                ResponseCandidate(
                    text=result,
                    confidence=confidence,
                    source=f"{self._source_name}:dspy",
                    safety_passed=safe,
                    confidence_components=components,
                    safety_metadata=safety_metadata,
                    source_metadata=self._build_grounded_metadata(
                        source=f"{self._source_name}:dspy",
                        confidence=confidence,
                        confidence_components=components,
                        safety_passed=safe,
                        safety_metadata=safety_metadata,
                        prompt=prompt,
                        response_mode="dspy",
                    ),
                )
            ]

        materialized = await self._backend.generate_candidates(prompt, self._num_candidates)
        candidates: list[ResponseCandidate] = []
        for item in materialized[: self._num_candidates]:
            text = item.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            confidence, components = self._calibrate_confidence(item, text)
            safe, safety_metadata = self._evaluate_safety(text)
            item_source = item.get("source") if isinstance(item.get("source"), str) else "sampling"
            source = f"{self._source_name}:{self._backend_name}:{item_source}"
            candidates.append(
                ResponseCandidate(
                    text=text,
                    confidence=confidence,
                    source=source,
                    safety_passed=safe,
                    confidence_components=components,
                    safety_metadata=safety_metadata,
                    source_metadata=self._build_grounded_metadata(
                        source=source,
                        confidence=confidence,
                        confidence_components=components,
                        safety_passed=safe,
                        safety_metadata=safety_metadata,
                        prompt=prompt,
                    ),
                )
            )

        if not candidates:
            raise ValueError("Generate response did not include valid candidates")
        return candidates

    async def _generate(self, prompt: str) -> str:
        candidates = await self._generate_candidates(prompt)
        return candidates[0].text

    async def _handle_context_event(self, msg: Msg) -> None:
        input_id = "unknown"
        try:
            data = json.loads(msg.data.decode())
            if not isinstance(data, dict):
                raise ValueError("ContextAssembled payload must be a dict")
            decoded_payload, envelope_meta = decode_payload_or_envelope(EventSubjects.CONTEXT_ASSEMBLED, data)
            data = decoded_payload
            trace_id = envelope_meta.get("trace_id") if isinstance(envelope_meta.get("trace_id"), str) else None
            causation_id = envelope_meta.get("event_id") if isinstance(envelope_meta.get("event_id"), str) else None
            if "retrieved_facts" in data:
                payload = ContextAssembledPayload.from_dict(data)
                input_id = payload.input_id
                user_input = _normalized_optional_text(payload.user_input)
                author_id = payload.author_id
                user_id = payload.user_id
                channel_id = payload.channel_id
                author_name = _normalized_optional_text(payload.author_name)
                channel_context = _normalized_optional_text(payload.channel_context)
                recent_turn_summary = _normalized_optional_text(payload.recent_turn_summary)
                facts = [str(item) for item in payload.retrieved_facts]
                multimodal_interpretations = (
                    payload.multimodal_interpretations
                    if isinstance(payload.multimodal_interpretations, dict)
                    else {}
                )
            else:
                input_id = data.get("input_id")
                author_id = data.get("author_id") if isinstance(data.get("author_id"), str) else None
                user_id = data.get("user_id") if isinstance(data.get("user_id"), str) else None
                channel_id = data.get("channel_id") if isinstance(data.get("channel_id"), str) else None
                user_input = _normalized_optional_text(data.get("user_input"))
                author_name = _normalized_optional_text(data.get("author_name"))
                channel_context = _normalized_optional_text(data.get("channel_context"))
                recent_turn_summary = _normalized_optional_text(data.get("recent_turn_summary"))
                facts = _extract_memory_facts(data)
                multimodal_interpretations = (
                    data.get("multimodal_interpretations")
                    if isinstance(data.get("multimodal_interpretations"), dict)
                    else {}
                )

            if author_id is None:
                author_id = user_id
            if user_input is None:
                raise ValueError("ContextAssembled payload missing required non-empty user_input")
            prompt = _build_generation_prompt(
                user_input=user_input,
                facts=facts,
                author_name=author_name,
                channel_context=channel_context,
                recent_turn_summary=recent_turn_summary,
                multimodal_interpretations=multimodal_interpretations,
            )

            fallback = multimodal_interpretations.get("fallback") if isinstance(multimodal_interpretations.get("fallback"), dict) else {}
            should_clarify = bool(fallback.get("ask_clarifying_question"))
            confidence_obj = multimodal_interpretations.get("confidence")
            if isinstance(confidence_obj, dict):
                should_clarify = should_clarify or bool(confidence_obj.get("low_confidence"))

            logger.info("RemoteLLM generating for %s", input_id)
            if should_clarify:
                candidates = [
                    ResponseCandidate(
                        text=_build_clarifying_question(multimodal_interpretations),
                        confidence=0.95,
                        source=f"{self._source_name}:clarifying_fallback",
                        safety_passed=True,
                        confidence_components={"fallback": 1.0},
                        safety_metadata={"rule": "low_multimodal_confidence", "safety_passed": True},
                        source_metadata=self._build_grounded_metadata(
                            source=f"{self._source_name}:clarifying_fallback",
                            confidence=0.95,
                            confidence_components={"fallback": 1.0},
                            safety_passed=True,
                            safety_metadata={"rule": "low_multimodal_confidence", "safety_passed": True},
                            prompt=prompt,
                            response_mode="clarifying_fallback",
                        ),
                    )
                ]
            else:
                candidates = await self._generate_candidates(prompt)
            payload = ResponseCandidatesPayload(
                candidates=candidates,
                input_id=input_id,
                user_id=author_id or user_id,
                author_id=author_id,
                channel_id=channel_id,
                timestamp=None,
            )
            envelope = EventEnvelope.build(
                subject=EventSubjects.RESPONSE_CANDIDATES,
                payload=json.loads(payload.to_json()),
                producer=self.__class__.__name__,
                trace_id=trace_id,
                causation_id=causation_id or input_id,
            )
            await self._publisher.publish(
                EventSubjects.RESPONSE_CANDIDATES,
                envelope.__dict__,
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
                subject=EventSubjects.CONTEXT_ASSEMBLED,
                handler=self._handle_context_event,
                use_jetstream=True,
                durable=durable_name,
            )
            logger.info("RemoteLLM subscribed to %s", EventSubjects.CONTEXT_ASSEMBLED)
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
        await self._backend.close()
        if not self._session.closed:
            await self._session.close()
        logger.info("RemoteLLM stopped listening.")
