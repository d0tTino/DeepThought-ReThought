import asyncio
import json

import pytest

from deepthought.eda.events import EventSubjects
from deepthought.services.context_assembler_service import ContextAssemblerService


class DummyNATS:
    is_connected = True


class DummyJS:
    pass


class RecordingPublisher:
    def __init__(self, *args, **kwargs):
        self.calls = []

    async def publish(self, subject, payload, use_jetstream=True, timeout=10.0):
        self.calls.append((subject, payload, use_jetstream, timeout))


class RecordingSubscriber:
    def __init__(self, *args, **kwargs):
        self.calls = []

    async def subscribe(self, **kwargs):
        self.calls.append(kwargs)
        return True

    async def unsubscribe_all(self):
        return None


class DummyMsg:
    def __init__(self, payload: dict):
        self.data = json.dumps(payload).encode()
        self.acked = False
        self.nacked = False

    async def ack(self):
        self.acked = True

    async def nak(self):
        self.nacked = True


class DummyDB:
    def __init__(self, adaptation_state=None):
        self.adaptation_state = adaptation_state or {}

    async def get_adaptation_state(self, *, user_id=None):
        return self.adaptation_state


@pytest.mark.asyncio
async def test_race_safe_assembly_collects_all_providers(monkeypatch):
    import deepthought.services.context_assembler_service as mod

    monkeypatch.setattr(mod, "Publisher", RecordingPublisher)
    monkeypatch.setattr(mod, "Subscriber", RecordingSubscriber)

    svc = ContextAssemblerService(DummyNATS(), DummyJS(), wait_window_seconds=0.12)

    input_msg = DummyMsg(
        {
            "input_id": "i-race",
            "user_input": "hello",
            "author_id": "u1",
            "conversation_window": [{"role": "user", "text": "earlier"}],
        }
    )
    await svc._handle_input_received(input_msg)

    async def send_memory():
        await asyncio.sleep(0.01)
        await svc._handle_provider_response(
            DummyMsg(
                {"input_id": "i-race", "retrieved_knowledge": {"facts": ["f2", "f1"]}}
            ),
            "memory",
        )

    async def send_social():
        await asyncio.sleep(0.03)
        await svc._handle_provider_response(
            DummyMsg({"input_id": "i-race", "social_signals": {"tone": "neutral"}}),
            "social",
        )

    async def send_perception():
        await asyncio.sleep(0.02)
        await svc._handle_provider_response(
            DummyMsg(
                {
                    "input_id": "i-race",
                    "multimodal_interpretations": {
                        "summary": "image processed",
                        "notes": ["ok"],
                        "by_modality": {"image": {"what": "none"}},
                    },
                }
            ),
            "perception",
        )

    await asyncio.gather(send_social(), send_memory(), send_perception())
    await asyncio.sleep(0.06)

    assembled = [
        c for c in svc._publisher.calls if c[0] == EventSubjects.CONTEXT_ASSEMBLED
    ]
    assert len(assembled) == 1
    payload = assembled[0][1]["payload"]
    assert payload["confidence"]["partial"] is False
    assert payload["confidence"]["required_providers"] == ["memory", "social"]
    assert payload["confidence"]["completed_providers"] == ["memory", "social"]
    assert payload["confidence"]["provider_missing_reasons"] == {}
    assert payload["confidence"]["provider_latency_p95_ms"]["memory"] is not None


@pytest.mark.asyncio
async def test_context_assembler_composes_request_and_memory_context(monkeypatch):
    import deepthought.services.context_assembler_service as mod

    monkeypatch.setattr(mod, "Publisher", RecordingPublisher)
    monkeypatch.setattr(mod, "Subscriber", RecordingSubscriber)

    svc = ContextAssemblerService(DummyNATS(), DummyJS(), wait_window_seconds=0.05)

    await svc._handle_input_received(
        DummyMsg(
            {
                "input_id": "i-compose",
                "user_input": "hello",
                "conversation_window": [
                    {
                        "role": "user",
                        "text": "from-request",
                        "timestamp": "2026-03-18T00:00:00+00:00",
                    }
                ],
            }
        )
    )
    await svc._handle_provider_response(
        DummyMsg(
            {
                "input_id": "i-compose",
                "retrieved_knowledge": {
                    "facts": ["fact-a"],
                    "conversation_window": [
                        {
                            "role": "assistant",
                            "text": "from-memory",
                            "timestamp": "2026-03-18T00:00:01+00:00",
                        }
                    ],
                    "recent_turn_summary": "summary-from-memory",
                    "layers": {"recent_episodic_turns": ["user: from-request"]},
                    "retrieval_policy": {"recent_turns": 4},
                },
            }
        ),
        "memory",
    )
    await svc._handle_provider_response(
        DummyMsg({"input_id": "i-compose", "social_signals": {"tone": "neutral"}}),
        "social",
    )
    await svc._handle_provider_response(
        DummyMsg(
            {
                "input_id": "i-compose",
                "multimodal_interpretations": {"summary": "none", "notes": []},
            }
        ),
        "perception",
    )
    await asyncio.sleep(0.06)

    assembled = [
        c for c in svc._publisher.calls if c[0] == EventSubjects.CONTEXT_ASSEMBLED
    ]
    payload = assembled[0][1]["payload"]
    assert [turn["text"] for turn in payload["conversation_window"]] == [
        "from-request",
        "from-memory",
    ]
    assert payload["recent_turn_summary"] == "summary-from-memory"
    assert payload["confidence"]["retrieval_layers"] == {
        "recent_episodic_turns": ["user: from-request"]
    }
    assert payload["confidence"]["retrieval_policy"] == {"recent_turns": 4}


@pytest.mark.asyncio
async def test_partial_result_uses_missing_reason_timeout(monkeypatch):
    import deepthought.services.context_assembler_service as mod

    monkeypatch.setattr(mod, "Publisher", RecordingPublisher)
    monkeypatch.setattr(mod, "Subscriber", RecordingSubscriber)

    svc = ContextAssemblerService(DummyNATS(), DummyJS(), wait_window_seconds=0.05)

    input_msg = DummyMsg({"input_id": "i-partial", "user_input": "hello"})
    await svc._handle_input_received(input_msg)

    await svc._handle_provider_response(
        DummyMsg({"input_id": "i-partial", "retrieved_knowledge": {"facts": ["f1"]}}),
        "memory",
    )
    await svc._handle_provider_response(
        DummyMsg({"input_id": "i-partial", "social_signals": {"sentiment": 0.7}}),
        "social",
    )

    await asyncio.sleep(0.09)

    assembled = [
        c for c in svc._publisher.calls if c[0] == EventSubjects.CONTEXT_ASSEMBLED
    ]
    assert len(assembled) == 1
    payload = assembled[0][1]["payload"]
    assert payload["confidence"]["missing_providers"] == []
    assert payload["confidence"]["provider_missing_reasons"] == {}
    assert payload["confidence"]["required_providers"] == ["memory", "social"]


@pytest.mark.asyncio
async def test_context_assembler_correlates_by_input_and_trace_id(monkeypatch):
    import deepthought.services.context_assembler_service as mod

    monkeypatch.setattr(mod, "Publisher", RecordingPublisher)
    monkeypatch.setattr(mod, "Subscriber", RecordingSubscriber)

    svc = ContextAssemblerService(DummyNATS(), DummyJS(), wait_window_seconds=0.04)

    await svc._handle_input_received(
        DummyMsg({"input_id": "i-trace", "user_input": "hello", "trace_id": "trace-A"})
    )

    await svc._handle_provider_response(
        DummyMsg(
            {
                "input_id": "i-trace",
                "trace_id": "trace-B",
                "retrieved_knowledge": {"facts": ["wrong"]},
            }
        ),
        "memory",
    )
    await svc._handle_provider_response(
        DummyMsg(
            {
                "input_id": "i-trace",
                "trace_id": "trace-A",
                "social_signals": {"sentiment": 0.9},
            }
        ),
        "social",
    )

    await asyncio.sleep(0.07)

    assembled = [
        c for c in svc._publisher.calls if c[0] == EventSubjects.CONTEXT_ASSEMBLED
    ]
    payload = assembled[0][1]["payload"]
    assert payload["retrieved_facts"] == []
    assert payload["confidence"]["provider_missing_reasons"]["memory"] == "timeout"


@pytest.mark.asyncio
async def test_late_arrival_emits_context_update(monkeypatch):
    import deepthought.services.context_assembler_service as mod

    monkeypatch.setattr(mod, "Publisher", RecordingPublisher)
    monkeypatch.setattr(mod, "Subscriber", RecordingSubscriber)

    svc = ContextAssemblerService(
        DummyNATS(),
        DummyJS(),
        wait_window_seconds=0.03,
        late_arrival_window_seconds=0.08,
    )

    await svc._handle_input_received(
        DummyMsg({"input_id": "i-late", "user_input": "hello"})
    )
    await svc._handle_provider_response(
        DummyMsg({"input_id": "i-late", "social_signals": {"tone": "ok"}}), "social"
    )
    await svc._handle_provider_response(
        DummyMsg(
            {
                "input_id": "i-late",
                "multimodal_interpretations": {"summary": "none", "notes": []},
            }
        ),
        "perception",
    )

    await asyncio.sleep(0.05)
    await svc._handle_provider_response(
        DummyMsg(
            {"input_id": "i-late", "retrieved_knowledge": {"facts": ["late-fact"]}}
        ),
        "memory",
    )
    await asyncio.sleep(0.01)

    assembled = [
        c for c in svc._publisher.calls if c[0] == EventSubjects.CONTEXT_ASSEMBLED
    ]
    updates = [c for c in svc._publisher.calls if c[0] == EventSubjects.CONTEXT_UPDATED]
    assert len(assembled) == 1
    assert len(updates) == 1
    update_payload = updates[0][1]["payload"]
    assert update_payload["retrieved_facts"] == ["late-fact"]
    assert update_payload["confidence"]["late_update"] is True
    assert update_payload["confidence"]["update_reason"] == "late_provider_merge"
    assert update_payload["confidence"]["missing_providers"] == []
    assert update_payload["confidence"]["quality_score"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_adaptive_deadline_uses_rolling_p95(monkeypatch):
    import deepthought.services.context_assembler_service as mod

    monkeypatch.setattr(mod, "Publisher", RecordingPublisher)
    monkeypatch.setattr(mod, "Subscriber", RecordingSubscriber)

    svc = ContextAssemblerService(
        DummyNATS(),
        DummyJS(),
        wait_window_seconds=0.02,
        provider_jitter_budget_seconds=0.005,
        late_arrival_window_seconds=0.0,
    )

    # Warm-up round with slower memory arrival to seed rolling latency history.
    await svc._handle_input_received(
        DummyMsg({"input_id": "i-seed", "user_input": "hello"})
    )
    await svc._handle_provider_response(
        DummyMsg({"input_id": "i-seed", "social_signals": {"tone": "ok"}}), "social"
    )
    await svc._handle_provider_response(
        DummyMsg(
            {
                "input_id": "i-seed",
                "multimodal_interpretations": {"summary": "ok", "notes": ["x"]},
            }
        ),
        "perception",
    )
    await asyncio.sleep(0.03)
    await svc._handle_provider_response(
        DummyMsg({"input_id": "i-seed", "retrieved_knowledge": {"facts": ["seed"]}}),
        "memory",
    )
    await asyncio.sleep(0.02)

    await svc._handle_input_received(
        DummyMsg({"input_id": "i-adaptive", "user_input": "hello2"})
    )
    pending = svc._pending["i-adaptive"]
    memory_budget = pending.provider_deadlines["memory"] - pending.started_at
    social_budget = pending.provider_deadlines["social"] - pending.started_at
    assert memory_budget > social_budget


@pytest.mark.asyncio
async def test_context_assembler_uses_social_signals_retrieved_as_social_provider(
    monkeypatch,
):
    import deepthought.services.context_assembler_service as mod

    monkeypatch.setattr(mod, "Publisher", RecordingPublisher)
    monkeypatch.setattr(mod, "Subscriber", RecordingSubscriber)

    svc = ContextAssemblerService(DummyNATS(), DummyJS(), wait_window_seconds=0.05)
    started = await svc.start(durable_name="ctx-social-contract")

    assert started is True
    subjects = [call["subject"] for call in svc._subscriber.calls]
    assert EventSubjects.SOCIAL_SIGNALS_RETRIEVED in subjects
    assert EventSubjects.SOCIAL_UPDATED not in subjects


@pytest.mark.asyncio
async def test_context_assembler_merges_adaptation_state_into_selector_inputs(
    monkeypatch,
):
    import deepthought.services.context_assembler_service as mod

    monkeypatch.setattr(mod, "Publisher", RecordingPublisher)
    monkeypatch.setattr(mod, "Subscriber", RecordingSubscriber)

    svc = ContextAssemblerService(
        DummyNATS(),
        DummyJS(),
        db_manager=DummyDB(
            {
                "user": {
                    "response_style": {"preferred": "concise"},
                    "fallback": {"aggressiveness": 0.2},
                    "memory": {"salience_boost": 0.7, "retrieval_priority": 0.9},
                },
                "sources": {
                    "responder:factual": {"selector": {"weight_multiplier": 1.25}},
                },
                "retrieval": {"salience_boost": 0.7, "retrieval_priority": 0.9},
                "fallback": {"aggressiveness": 0.2},
            }
        ),
        wait_window_seconds=0.05,
    )

    await svc._handle_input_received(
        DummyMsg({"input_id": "i-adapt", "user_input": "hello", "author_id": "u-adapt"})
    )
    await svc._handle_provider_response(
        DummyMsg({"input_id": "i-adapt", "retrieved_knowledge": {"facts": ["fact-a"]}}),
        "memory",
    )
    await svc._handle_provider_response(
        DummyMsg(
            {
                "input_id": "i-adapt",
                "social_signals": {
                    "selector_inputs": {
                        "interaction_policy": {"ask_clarifying_on_no_safe": True}
                    }
                },
            }
        ),
        "social",
    )
    await svc._handle_provider_response(
        DummyMsg(
            {
                "input_id": "i-adapt",
                "multimodal_interpretations": {"summary": "none", "notes": []},
            }
        ),
        "perception",
    )
    await asyncio.sleep(0.06)

    assembled = [
        c for c in svc._publisher.calls if c[0] == EventSubjects.CONTEXT_ASSEMBLED
    ]
    payload = assembled[0][1]["payload"]
    assert (
        payload["adaptation_state"]["user"]["response_style"]["preferred"] == "concise"
    )
    assert (
        payload["social_signals"]["selector_inputs"]["interaction_policy"][
            "response_style"
        ]
        == "concise"
    )
    assert payload["social_signals"]["selector_inputs"]["interaction_policy"][
        "fallback_aggressiveness"
    ] == pytest.approx(0.2)
    assert payload["confidence"]["retrieval_policy"]["salience_boost"] == pytest.approx(
        0.7
    )
    assert payload["social_signals"]["selector_inputs"]["user_history_affinity"][
        "responder:factual"
    ] == pytest.approx(0.25)


@pytest.mark.asyncio
async def test_qos_profile_multimodal_sets_required_providers(monkeypatch):
    import deepthought.services.context_assembler_service as mod

    monkeypatch.setattr(mod, "Publisher", RecordingPublisher)
    monkeypatch.setattr(mod, "Subscriber", RecordingSubscriber)

    svc = ContextAssemblerService(DummyNATS(), DummyJS(), wait_window_seconds=0.04)
    await svc._handle_input_received(
        DummyMsg(
            {
                "input_id": "i-profile-multi",
                "user_input": "what is in this image?",
                "attachments": [
                    {"type": "image", "url": "https://example.com/image.png"}
                ],
            }
        )
    )
    await svc._handle_provider_response(
        DummyMsg(
            {"input_id": "i-profile-multi", "retrieved_knowledge": {"facts": ["f1"]}}
        ),
        "memory",
    )
    await svc._handle_provider_response(
        DummyMsg(
            {
                "input_id": "i-profile-multi",
                "multimodal_interpretations": {"summary": "cat", "notes": ["cat"]},
            }
        ),
        "perception",
    )
    await asyncio.sleep(0.06)

    assembled = [
        c for c in svc._publisher.calls if c[0] == EventSubjects.CONTEXT_ASSEMBLED
    ]
    payload = assembled[0][1]["payload"]
    assert payload["confidence"]["qos_profile"] == "multimodal"
    assert payload["confidence"]["required_providers"] == ["memory", "perception"]
    assert payload["confidence"]["missing_providers"] == []
    assert payload["confidence"]["quality_score"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_qos_profile_high_risk_requires_all_providers(monkeypatch):
    import deepthought.services.context_assembler_service as mod

    monkeypatch.setattr(mod, "Publisher", RecordingPublisher)
    monkeypatch.setattr(mod, "Subscriber", RecordingSubscriber)

    svc = ContextAssemblerService(DummyNATS(), DummyJS(), wait_window_seconds=0.03)
    await svc._handle_input_received(
        DummyMsg(
            {
                "input_id": "i-profile-risk",
                "user_input": "I have an emergency and might self-harm",
            }
        )
    )
    await svc._handle_provider_response(
        DummyMsg(
            {
                "input_id": "i-profile-risk",
                "retrieved_knowledge": {"facts": ["hotline"]},
            }
        ),
        "memory",
    )
    await svc._handle_provider_response(
        DummyMsg({"input_id": "i-profile-risk", "social_signals": {"risk": "high"}}),
        "social",
    )
    await asyncio.sleep(0.07)

    assembled = [
        c for c in svc._publisher.calls if c[0] == EventSubjects.CONTEXT_ASSEMBLED
    ]
    payload = assembled[0][1]["payload"]
    assert payload["confidence"]["qos_profile"] == "high_risk"
    assert payload["confidence"]["required_providers"] == [
        "memory",
        "social",
        "perception",
    ]
    assert payload["confidence"]["missing_reasons"]["perception"] == "timeout"
    assert payload["confidence"]["quality_score"] < 1.0
