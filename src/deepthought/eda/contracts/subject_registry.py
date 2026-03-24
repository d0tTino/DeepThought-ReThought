"""Strict subject registry with lifecycle metadata for DeepThought EDA subjects."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Final

logger = logging.getLogger(__name__)


class SubjectLifecycleState(str, Enum):
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    REMOVE_BY_DATE = "remove-by-date"


@dataclass(frozen=True)
class SubjectMetadata:
    alias: str
    canonical: str
    schema_id: str
    lifecycle: SubjectLifecycleState = SubjectLifecycleState.ACTIVE
    remove_by: str | None = None


class SubjectAliases:
    INPUT_RECEIVED: Final[str] = "dtr.input.received"
    MEMORY_RETRIEVED: Final[str] = "dtr.memory.retrieved"
    MEMORY_RETRIEVAL_REQUESTED: Final[str] = "dtr.memory.retrieval.requested"
    SOCIAL_SIGNALS_REQUESTED: Final[str] = "dtr.social.signals.requested"
    PERCEPTION_INTERPRET_REQUESTED: Final[str] = "dtr.perception.interpret.requested"
    SOCIAL_SIGNALS_RETRIEVED: Final[str] = "dtr.social.signals.retrieved"
    PERCEPTION_INTERPRET_RETRIEVED: Final[str] = "dtr.perception.interpret.retrieved"
    CONTEXT_ASSEMBLED: Final[str] = "dtr.context.assembled"
    CONTEXT_UPDATED: Final[str] = "dtr.context.updated"
    RESPONSE_CANDIDATES: Final[str] = "dtr.response.candidates"
    RESPONSE_RANKED: Final[str] = "dtr.response.ranked"
    RESPONSE_GENERATED: Final[str] = "dtr.llm.response_generated"
    PERCEPTION_EMBEDDINGS: Final[str] = "dtr.perception.embeddings"
    PERCEPTION_EXTRACT: Final[str] = "dtr.perception.extract"
    PERCEPTION_EXTRACT_REQUESTED: Final[str] = "dtr.perception.extract.requested"
    PERCEPTION_MODALITY_RESULT: Final[str] = "dtr.perception.modality.result"
    SOCIAL_PERCEPTION: Final[str] = "dtr.social.perception"
    OUTCOME_SIGNAL: Final[str] = "dtr.feedback.outcome_signal"
    CORRECTION_SIGNAL: Final[str] = "dtr.feedback.correction_signal"
    DISCORD_FEEDBACK_SIGNAL: Final[str] = "dtr.feedback.discord_signal"
    USER_SUMMARY_REFRESH: Final[str] = "dtr.memory.user_summary.refresh"

    SOCIAL_UPDATED: Final[str] = "dtr.social.updated"
    PERCEPTION_IMAGE_EMBED: Final[str] = "dtr.perception.image_embeddings"
    PERCEPTION_AUDIO_EMBED: Final[str] = "dtr.perception.audio_embeddings"
    PERCEPTION_VIDEO_EMBED: Final[str] = "dtr.perception.video_embeddings"
    REMINDER_TRIGGERED: Final[str] = "dtr.scheduler.reminder_triggered"
    MICRO_TICK: Final[str] = "dtr.scheduler.micro_tick"
    DAILY_STANDUP: Final[str] = "dtr.scheduler.daily_standup"
    WEEKLY_PLANNING: Final[str] = "dtr.scheduler.weekly_planning"
    CODE_TEMPLATE_REQUEST: Final[str] = "dtr.codegen.template_request"
    CODE_GENERATED: Final[str] = "dtr.codegen.generated"
    PLAN_REQUESTED: Final[str] = "dtr.plan.requested"
    PLAN_GENERATED: Final[str] = "dtr.plan.generated"
    BDI_INTENTION: Final[str] = "dtr.bdi.intention"
    WARNING: Final[str] = "dtr.warning"

    TELEMETRY_SELECTOR_RANKING: Final[str] = "dtr.telemetry.selector_ranking.v1"
    TELEMETRY_EGRESS_POLICY: Final[str] = "dtr.telemetry.egress_policy.v1"
    TELEMETRY_RESPONSE_FEEDBACK: Final[str] = "dtr.telemetry.response_feedback.v1"
    TRAINING_FEEDBACK_TUPLES: Final[str] = "dtr.training.feedback_tuples.v1"


class SubjectCanonicals:
    INPUT_RECEIVED: Final[str] = "dtr.input.received.v1"
    MEMORY_RETRIEVED: Final[str] = "dtr.memory.retrieved.v1"
    MEMORY_RETRIEVAL_REQUESTED: Final[str] = "dtr.memory.retrieval.requested.v1"
    SOCIAL_SIGNALS_REQUESTED: Final[str] = "dtr.social.signals.requested.v1"
    PERCEPTION_INTERPRET_REQUESTED: Final[str] = "dtr.perception.interpret.requested.v1"
    SOCIAL_SIGNALS_RETRIEVED: Final[str] = "dtr.social.signals.retrieved.v1"
    PERCEPTION_INTERPRET_RETRIEVED: Final[str] = "dtr.perception.interpret.retrieved.v1"
    CONTEXT_ASSEMBLED: Final[str] = "dtr.context.assembled.v1"
    CONTEXT_UPDATED: Final[str] = "dtr.context.updated.v1"
    RESPONSE_CANDIDATES: Final[str] = "dtr.response.candidates.v1"
    RESPONSE_RANKED: Final[str] = "dtr.response.ranked.v1"
    PERCEPTION_EMBEDDINGS: Final[str] = "dtr.perception.embeddings.v1"
    PERCEPTION_EXTRACT: Final[str] = "dtr.perception.extract.v1"
    PERCEPTION_EXTRACT_REQUESTED: Final[str] = "dtr.perception.extract.requested.v1"
    PERCEPTION_MODALITY_RESULT: Final[str] = "dtr.perception.modality.result.v1"
    SOCIAL_PERCEPTION: Final[str] = "dtr.social.perception.v1"
    OUTCOME_SIGNAL: Final[str] = "dtr.feedback.outcome_signal.v1"
    CORRECTION_SIGNAL: Final[str] = "dtr.feedback.correction_signal.v1"
    DISCORD_FEEDBACK_SIGNAL: Final[str] = "dtr.feedback.discord_signal.v1"
    USER_SUMMARY_REFRESH: Final[str] = "dtr.memory.user_summary.refresh.v1"


SUBJECT_REGISTRY: dict[str, SubjectMetadata] = {
    SubjectCanonicals.INPUT_RECEIVED: SubjectMetadata(SubjectCanonicals.INPUT_RECEIVED, SubjectCanonicals.INPUT_RECEIVED, "schemas/eda/input_received.v1.json"),
    SubjectAliases.INPUT_RECEIVED: SubjectMetadata(SubjectAliases.INPUT_RECEIVED, SubjectCanonicals.INPUT_RECEIVED, "schemas/eda/input_received.v1.json", SubjectLifecycleState.DEPRECATED),
    SubjectCanonicals.MEMORY_RETRIEVED: SubjectMetadata(SubjectCanonicals.MEMORY_RETRIEVED, SubjectCanonicals.MEMORY_RETRIEVED, "schemas/eda/memory_retrieved.v1.json"),
    SubjectAliases.MEMORY_RETRIEVED: SubjectMetadata(SubjectAliases.MEMORY_RETRIEVED, SubjectCanonicals.MEMORY_RETRIEVED, "schemas/eda/memory_retrieved.v1.json", SubjectLifecycleState.DEPRECATED),
    SubjectCanonicals.MEMORY_RETRIEVAL_REQUESTED: SubjectMetadata(SubjectCanonicals.MEMORY_RETRIEVAL_REQUESTED, SubjectCanonicals.MEMORY_RETRIEVAL_REQUESTED, "schemas/eda/memory_retrieval_requested.v1.json"),
    SubjectAliases.MEMORY_RETRIEVAL_REQUESTED: SubjectMetadata(SubjectAliases.MEMORY_RETRIEVAL_REQUESTED, SubjectCanonicals.MEMORY_RETRIEVAL_REQUESTED, "schemas/eda/memory_retrieval_requested.v1.json", SubjectLifecycleState.DEPRECATED),
    SubjectCanonicals.SOCIAL_SIGNALS_REQUESTED: SubjectMetadata(SubjectCanonicals.SOCIAL_SIGNALS_REQUESTED, SubjectCanonicals.SOCIAL_SIGNALS_REQUESTED, "schemas/eda/social_signals_requested.v1.json"),
    SubjectAliases.SOCIAL_SIGNALS_REQUESTED: SubjectMetadata(SubjectAliases.SOCIAL_SIGNALS_REQUESTED, SubjectCanonicals.SOCIAL_SIGNALS_REQUESTED, "schemas/eda/social_signals_requested.v1.json", SubjectLifecycleState.DEPRECATED),
    SubjectCanonicals.PERCEPTION_INTERPRET_REQUESTED: SubjectMetadata(SubjectCanonicals.PERCEPTION_INTERPRET_REQUESTED, SubjectCanonicals.PERCEPTION_INTERPRET_REQUESTED, "schemas/eda/perception_interpret_requested.v1.json"),
    SubjectAliases.PERCEPTION_INTERPRET_REQUESTED: SubjectMetadata(SubjectAliases.PERCEPTION_INTERPRET_REQUESTED, SubjectCanonicals.PERCEPTION_INTERPRET_REQUESTED, "schemas/eda/perception_interpret_requested.v1.json", SubjectLifecycleState.DEPRECATED),
    SubjectCanonicals.SOCIAL_SIGNALS_RETRIEVED: SubjectMetadata(SubjectCanonicals.SOCIAL_SIGNALS_RETRIEVED, SubjectCanonicals.SOCIAL_SIGNALS_RETRIEVED, "schemas/eda/social_signals_retrieved.v1.json"),
    SubjectAliases.SOCIAL_SIGNALS_RETRIEVED: SubjectMetadata(SubjectAliases.SOCIAL_SIGNALS_RETRIEVED, SubjectCanonicals.SOCIAL_SIGNALS_RETRIEVED, "schemas/eda/social_signals_retrieved.v1.json", SubjectLifecycleState.DEPRECATED),
    SubjectCanonicals.PERCEPTION_INTERPRET_RETRIEVED: SubjectMetadata(SubjectCanonicals.PERCEPTION_INTERPRET_RETRIEVED, SubjectCanonicals.PERCEPTION_INTERPRET_RETRIEVED, "schemas/eda/perception_interpret_retrieved.v1.json"),
    SubjectAliases.PERCEPTION_INTERPRET_RETRIEVED: SubjectMetadata(SubjectAliases.PERCEPTION_INTERPRET_RETRIEVED, SubjectCanonicals.PERCEPTION_INTERPRET_RETRIEVED, "schemas/eda/perception_interpret_retrieved.v1.json", SubjectLifecycleState.DEPRECATED),
    SubjectCanonicals.CONTEXT_ASSEMBLED: SubjectMetadata(SubjectCanonicals.CONTEXT_ASSEMBLED, SubjectCanonicals.CONTEXT_ASSEMBLED, "schemas/eda/context_assembled.v1.json"),
    SubjectAliases.CONTEXT_ASSEMBLED: SubjectMetadata(SubjectAliases.CONTEXT_ASSEMBLED, SubjectCanonicals.CONTEXT_ASSEMBLED, "schemas/eda/context_assembled.v1.json", SubjectLifecycleState.DEPRECATED),
    SubjectCanonicals.CONTEXT_UPDATED: SubjectMetadata(SubjectCanonicals.CONTEXT_UPDATED, SubjectCanonicals.CONTEXT_UPDATED, "schemas/eda/context_updated.v1.json"),
    SubjectAliases.CONTEXT_UPDATED: SubjectMetadata(SubjectAliases.CONTEXT_UPDATED, SubjectCanonicals.CONTEXT_UPDATED, "schemas/eda/context_updated.v1.json", SubjectLifecycleState.DEPRECATED),
    SubjectCanonicals.RESPONSE_CANDIDATES: SubjectMetadata(SubjectCanonicals.RESPONSE_CANDIDATES, SubjectCanonicals.RESPONSE_CANDIDATES, "schemas/eda/response_candidates.v1.json"),
    SubjectAliases.RESPONSE_CANDIDATES: SubjectMetadata(SubjectAliases.RESPONSE_CANDIDATES, SubjectCanonicals.RESPONSE_CANDIDATES, "schemas/eda/response_candidates.v1.json", SubjectLifecycleState.DEPRECATED),
    SubjectCanonicals.RESPONSE_RANKED: SubjectMetadata(SubjectCanonicals.RESPONSE_RANKED, SubjectCanonicals.RESPONSE_RANKED, "schemas/eda/response_ranked.v1.json"),
    SubjectAliases.RESPONSE_RANKED: SubjectMetadata(SubjectAliases.RESPONSE_RANKED, SubjectCanonicals.RESPONSE_RANKED, "schemas/eda/response_ranked.v1.json", SubjectLifecycleState.DEPRECATED),
    SubjectAliases.RESPONSE_GENERATED: SubjectMetadata(SubjectAliases.RESPONSE_GENERATED, SubjectCanonicals.RESPONSE_RANKED, "schemas/eda/response_ranked.v1.json", SubjectLifecycleState.REMOVE_BY_DATE, remove_by="2026-12-31"),
    SubjectCanonicals.PERCEPTION_EMBEDDINGS: SubjectMetadata(SubjectCanonicals.PERCEPTION_EMBEDDINGS, SubjectCanonicals.PERCEPTION_EMBEDDINGS, "schemas/eda/perception_embeddings.v1.json"),
    SubjectAliases.PERCEPTION_EMBEDDINGS: SubjectMetadata(SubjectAliases.PERCEPTION_EMBEDDINGS, SubjectCanonicals.PERCEPTION_EMBEDDINGS, "schemas/eda/perception_embeddings.v1.json", SubjectLifecycleState.DEPRECATED),
    SubjectCanonicals.PERCEPTION_EXTRACT: SubjectMetadata(SubjectCanonicals.PERCEPTION_EXTRACT, SubjectCanonicals.PERCEPTION_EXTRACT, "schemas/eda/perception_extract.v1.json"),
    SubjectAliases.PERCEPTION_EXTRACT: SubjectMetadata(SubjectAliases.PERCEPTION_EXTRACT, SubjectCanonicals.PERCEPTION_EXTRACT, "schemas/eda/perception_extract.v1.json", SubjectLifecycleState.DEPRECATED),
    SubjectCanonicals.PERCEPTION_EXTRACT_REQUESTED: SubjectMetadata(SubjectCanonicals.PERCEPTION_EXTRACT_REQUESTED, SubjectCanonicals.PERCEPTION_EXTRACT_REQUESTED, "schemas/eda/perception_extract_requested.v1.json"),
    SubjectAliases.PERCEPTION_EXTRACT_REQUESTED: SubjectMetadata(SubjectAliases.PERCEPTION_EXTRACT_REQUESTED, SubjectCanonicals.PERCEPTION_EXTRACT_REQUESTED, "schemas/eda/perception_extract_requested.v1.json", SubjectLifecycleState.DEPRECATED),
    SubjectCanonicals.PERCEPTION_MODALITY_RESULT: SubjectMetadata(SubjectCanonicals.PERCEPTION_MODALITY_RESULT, SubjectCanonicals.PERCEPTION_MODALITY_RESULT, "schemas/eda/perception_modality_result.v1.json"),
    SubjectAliases.PERCEPTION_MODALITY_RESULT: SubjectMetadata(SubjectAliases.PERCEPTION_MODALITY_RESULT, SubjectCanonicals.PERCEPTION_MODALITY_RESULT, "schemas/eda/perception_modality_result.v1.json", SubjectLifecycleState.DEPRECATED),
    SubjectCanonicals.SOCIAL_PERCEPTION: SubjectMetadata(SubjectCanonicals.SOCIAL_PERCEPTION, SubjectCanonicals.SOCIAL_PERCEPTION, "schemas/eda/social_perception.v1.json"),
    SubjectAliases.SOCIAL_PERCEPTION: SubjectMetadata(SubjectAliases.SOCIAL_PERCEPTION, SubjectCanonicals.SOCIAL_PERCEPTION, "schemas/eda/social_perception.v1.json", SubjectLifecycleState.DEPRECATED),
    SubjectCanonicals.OUTCOME_SIGNAL: SubjectMetadata(SubjectCanonicals.OUTCOME_SIGNAL, SubjectCanonicals.OUTCOME_SIGNAL, "schemas/eda/outcome_signal.v1.json"),
    SubjectAliases.OUTCOME_SIGNAL: SubjectMetadata(SubjectAliases.OUTCOME_SIGNAL, SubjectCanonicals.OUTCOME_SIGNAL, "schemas/eda/outcome_signal.v1.json", SubjectLifecycleState.DEPRECATED),
    SubjectCanonicals.CORRECTION_SIGNAL: SubjectMetadata(SubjectCanonicals.CORRECTION_SIGNAL, SubjectCanonicals.CORRECTION_SIGNAL, "schemas/eda/correction_signal.v1.json"),
    SubjectAliases.CORRECTION_SIGNAL: SubjectMetadata(SubjectAliases.CORRECTION_SIGNAL, SubjectCanonicals.CORRECTION_SIGNAL, "schemas/eda/correction_signal.v1.json", SubjectLifecycleState.DEPRECATED),
    SubjectCanonicals.DISCORD_FEEDBACK_SIGNAL: SubjectMetadata(SubjectCanonicals.DISCORD_FEEDBACK_SIGNAL, SubjectCanonicals.DISCORD_FEEDBACK_SIGNAL, "schemas/eda/discord_feedback_signal.v1.json"),
    SubjectAliases.DISCORD_FEEDBACK_SIGNAL: SubjectMetadata(SubjectAliases.DISCORD_FEEDBACK_SIGNAL, SubjectCanonicals.DISCORD_FEEDBACK_SIGNAL, "schemas/eda/discord_feedback_signal.v1.json", SubjectLifecycleState.DEPRECATED),
    SubjectCanonicals.USER_SUMMARY_REFRESH: SubjectMetadata(SubjectCanonicals.USER_SUMMARY_REFRESH, SubjectCanonicals.USER_SUMMARY_REFRESH, "schemas/eda/user_summary_refresh.v1.json"),
    SubjectAliases.USER_SUMMARY_REFRESH: SubjectMetadata(SubjectAliases.USER_SUMMARY_REFRESH, SubjectCanonicals.USER_SUMMARY_REFRESH, "schemas/eda/user_summary_refresh.v1.json", SubjectLifecycleState.DEPRECATED),

    SubjectAliases.SOCIAL_UPDATED: SubjectMetadata(SubjectAliases.SOCIAL_UPDATED, SubjectAliases.SOCIAL_UPDATED, "schemas/eda/social_updated.v1.json"),
    SubjectAliases.PERCEPTION_IMAGE_EMBED: SubjectMetadata(SubjectAliases.PERCEPTION_IMAGE_EMBED, SubjectAliases.PERCEPTION_IMAGE_EMBED, "schemas/eda/perception_image_embed.v1.json"),
    SubjectAliases.PERCEPTION_AUDIO_EMBED: SubjectMetadata(SubjectAliases.PERCEPTION_AUDIO_EMBED, SubjectAliases.PERCEPTION_AUDIO_EMBED, "schemas/eda/perception_audio_embed.v1.json"),
    SubjectAliases.PERCEPTION_VIDEO_EMBED: SubjectMetadata(SubjectAliases.PERCEPTION_VIDEO_EMBED, SubjectAliases.PERCEPTION_VIDEO_EMBED, "schemas/eda/perception_video_embed.v1.json"),
    SubjectAliases.REMINDER_TRIGGERED: SubjectMetadata(SubjectAliases.REMINDER_TRIGGERED, SubjectAliases.REMINDER_TRIGGERED, "schemas/eda/scheduler_reminder_triggered.v1.json"),
    SubjectAliases.MICRO_TICK: SubjectMetadata(SubjectAliases.MICRO_TICK, SubjectAliases.MICRO_TICK, "schemas/eda/scheduler_micro_tick.v1.json"),
    SubjectAliases.DAILY_STANDUP: SubjectMetadata(SubjectAliases.DAILY_STANDUP, SubjectAliases.DAILY_STANDUP, "schemas/eda/scheduler_daily_standup.v1.json"),
    SubjectAliases.WEEKLY_PLANNING: SubjectMetadata(SubjectAliases.WEEKLY_PLANNING, SubjectAliases.WEEKLY_PLANNING, "schemas/eda/scheduler_weekly_planning.v1.json"),
    SubjectAliases.CODE_TEMPLATE_REQUEST: SubjectMetadata(SubjectAliases.CODE_TEMPLATE_REQUEST, SubjectAliases.CODE_TEMPLATE_REQUEST, "schemas/eda/code_template_request.v1.json"),
    SubjectAliases.CODE_GENERATED: SubjectMetadata(SubjectAliases.CODE_GENERATED, SubjectAliases.CODE_GENERATED, "schemas/eda/code_generated.v1.json"),
    SubjectAliases.PLAN_REQUESTED: SubjectMetadata(SubjectAliases.PLAN_REQUESTED, SubjectAliases.PLAN_REQUESTED, "schemas/eda/plan_requested.v1.json"),
    SubjectAliases.PLAN_GENERATED: SubjectMetadata(SubjectAliases.PLAN_GENERATED, SubjectAliases.PLAN_GENERATED, "schemas/eda/plan_generated.v1.json"),
    SubjectAliases.BDI_INTENTION: SubjectMetadata(SubjectAliases.BDI_INTENTION, SubjectAliases.BDI_INTENTION, "schemas/eda/bdi_intention.v1.json"),
    SubjectAliases.WARNING: SubjectMetadata(SubjectAliases.WARNING, SubjectAliases.WARNING, "schemas/eda/warning.v1.json"),
    SubjectAliases.TELEMETRY_SELECTOR_RANKING: SubjectMetadata(SubjectAliases.TELEMETRY_SELECTOR_RANKING, SubjectAliases.TELEMETRY_SELECTOR_RANKING, "schemas/eda/telemetry_selector_ranking.v1.json"),
    SubjectAliases.TELEMETRY_EGRESS_POLICY: SubjectMetadata(SubjectAliases.TELEMETRY_EGRESS_POLICY, SubjectAliases.TELEMETRY_EGRESS_POLICY, "schemas/eda/telemetry_egress_policy.v1.json"),
    SubjectAliases.TELEMETRY_RESPONSE_FEEDBACK: SubjectMetadata(SubjectAliases.TELEMETRY_RESPONSE_FEEDBACK, SubjectAliases.TELEMETRY_RESPONSE_FEEDBACK, "schemas/eda/telemetry_response_feedback.v1.json"),
    SubjectAliases.TRAINING_FEEDBACK_TUPLES: SubjectMetadata(SubjectAliases.TRAINING_FEEDBACK_TUPLES, SubjectAliases.TRAINING_FEEDBACK_TUPLES, "schemas/eda/training_feedback_tuples.v1.json"),
}

_WARNED_ALIASES: set[str] = set()


def resolve_subject(subject: str) -> str:
    metadata = SUBJECT_REGISTRY.get(subject)
    if metadata is None:
        return subject

    if metadata.lifecycle in (SubjectLifecycleState.DEPRECATED, SubjectLifecycleState.REMOVE_BY_DATE):
        if subject not in _WARNED_ALIASES:
            _WARNED_ALIASES.add(subject)
            extra = ""
            if metadata.lifecycle is SubjectLifecycleState.REMOVE_BY_DATE and metadata.remove_by:
                today = date.today().isoformat()
                extra = f"; remove by {metadata.remove_by} (today: {today})"
            logger.warning(
                "Deprecated EDA subject alias '%s' used; canonical subject is '%s'%s",
                subject,
                metadata.canonical,
                extra,
            )
    return metadata.canonical


def get_subject_metadata(subject: str) -> SubjectMetadata | None:
    return SUBJECT_REGISTRY.get(subject)


def legacy_subject_map() -> dict[str, str]:
    return {
        subject: meta.canonical
        for subject, meta in SUBJECT_REGISTRY.items()
        if subject != meta.canonical
    }
