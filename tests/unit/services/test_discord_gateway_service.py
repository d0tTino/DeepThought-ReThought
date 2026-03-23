import asyncio
import json
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from deepthought.eda.contracts import EventEnvelope
from deepthought.eda.events import EventSubjects, ResponseRankedPayload
from deepthought.services.discord_gateway_service import DiscordGatewayService


def run(coro):
    return asyncio.run(coro)


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    async def sleep(self, seconds: float):
        self.now += max(0.0, seconds)


class DummyNATS:
    is_connected = True


class DummyJS:
    pass


class DummyPublisher:
    def __init__(self, *args, **kwargs):
        self.published = []

    async def publish(self, subject, payload, use_jetstream=True, timeout=10.0):
        self.published.append((subject, payload, use_jetstream, timeout))


class DummySubscriber:
    def __init__(self, *args, **kwargs):
        pass

    async def subscribe(self, **kwargs):
        return True

    async def unsubscribe_all(self):
        return None


class DummyMsg:
    def __init__(self, data):
        self.data = data.encode()
        self.acked = False
        self.nacked = False
        self.ack_calls = 0
        self.nak_calls = 0

    async def ack(self):
        self.acked = True
        self.ack_calls += 1

    async def nak(self):
        self.nacked = True
        self.nak_calls += 1


class DummyTypingScope:
    def __init__(self, channel):
        self._channel = channel

    async def __aenter__(self):
        self._channel.typing_entries += 1

    async def __aexit__(self, exc_type, exc, tb):
        return None


@dataclass
class DummyMessageReference:
    message_id: int | None
    channel_id: int | None
    fail_if_not_exists: bool = False


class DummyPartialMessage:
    def __init__(self, channel, message_id: int):
        self.channel = channel
        self.id = message_id

    def to_reference(self, *, fail_if_not_exists: bool = False) -> DummyMessageReference:
        return DummyMessageReference(
            message_id=self.id,
            channel_id=self.channel.id,
            fail_if_not_exists=fail_if_not_exists,
        )


class DummyChannel:
    def __init__(self, channel_id: int, *, kind: str = "channel"):
        self.id = channel_id
        self.kind = kind
        self.messages = []
        self.typing_entries = 0
        self.partial_message_requests = []

    def typing(self):
        return DummyTypingScope(self)

    def get_partial_message(self, message_id: int):
        self.partial_message_requests.append(message_id)
        return DummyPartialMessage(self, message_id)

    async def send(self, content: str, **kwargs):
        self.messages.append((content, kwargs))


class DummyDiscordClient:
    def __init__(self):
        self.channel = DummyChannel(123)
        self.thread = DummyChannel(999, kind="thread")
        self._cache = {123: self.channel, 999: self.thread}
        self._fetchable = {123: self.channel, 999: self.thread}
        self.fetch_calls = []

    def get_channel(self, channel_id: int):
        return self._cache.get(channel_id)

    async def fetch_channel(self, channel_id: int):
        self.fetch_calls.append(channel_id)
        channel = self._fetchable.get(channel_id)
        if channel is None:
            raise LookupError(channel_id)
        self._cache[channel_id] = channel
        return channel


def build_service(monkeypatch):
    import deepthought.services.base as base_mod

    fake_clock = FakeClock()
    discord_client = DummyDiscordClient()
    monkeypatch.setattr(base_mod, "Publisher", DummyPublisher)
    monkeypatch.setattr(base_mod, "Subscriber", DummySubscriber)
    service = DiscordGatewayService(
        DummyNATS(),
        DummyJS(),
        discord_client=discord_client,
        clock=fake_clock,
        sleeper=fake_clock.sleep,
    )
    return service, fake_clock, discord_client


def test_handle_discord_message_publishes_input_received(monkeypatch):
    service, _fake_clock, _discord_client = build_service(monkeypatch)
    message = SimpleNamespace(
        content="hello",
        id=99,
        reference=SimpleNamespace(message_id=66),
        thread=SimpleNamespace(id=999),
        author=SimpleNamespace(id=5, name="alice", display_name="Alice", bot=False),
        channel=SimpleNamespace(id=123),
        guild=SimpleNamespace(id=7),
        attachments=[
            SimpleNamespace(
                url="https://cdn.discordapp.com/file.png",
                content_type="image/png",
                filename="file.png",
                size=512,
            )
        ],
    )

    input_id = run(service.handle_discord_message(message))

    assert input_id is not None
    subject, payload, use_js, _timeout = service._publisher.published[0]
    assert subject == EventSubjects.INPUT_RECEIVED
    assert use_js is True
    assert payload["payload"]["user_input"] == "hello"
    assert payload["payload"]["channel_id"] == "123"
    assert payload["payload"]["reference_message_id"] == "66"
    assert payload["payload"]["thread_id"] == "999"
    assert payload["payload"]["attachments"] is not None
    assert payload["payload"]["conversation_window"][-1]["text"] == "hello"
    assert payload["payload"]["attachments"][0]["url"] == "https://cdn.discordapp.com/file.png"
    extract_subject, extract_payload, _, _ = service._publisher.published[1]
    assert extract_subject == EventSubjects.PERCEPTION_EXTRACT_REQUESTED
    assert extract_payload["payload"]["input_id"] == input_id
    assert extract_payload["payload"]["attachments"][0]["content_type"] == "image/png"


def test_handle_discord_message_includes_bounded_recent_window(monkeypatch):
    service, _fake_clock, _discord_client = build_service(monkeypatch)
    for idx in range(8):
        message = SimpleNamespace(
            content=f"turn-{idx}",
            id=200 + idx,
            author=SimpleNamespace(id=5, name="alice", display_name="Alice", bot=False),
            channel=SimpleNamespace(id=123),
            guild=SimpleNamespace(id=7),
            thread=SimpleNamespace(id=999),
        )
        run(service.handle_discord_message(message))

    subject, payload, _, _ = service._publisher.published[-1]
    assert subject == EventSubjects.INPUT_RECEIVED
    window = payload["payload"]["conversation_window"]
    assert len(window) == 6
    assert [turn["text"] for turn in window] == [f"turn-{idx}" for idx in range(2, 8)]
    assert payload["payload"]["recent_turn_summary"] == "turn-4 | turn-5 | turn-6"


def test_handle_discord_message_ignores_bot_authored_messages(monkeypatch):
    service, _fake_clock, _discord_client = build_service(monkeypatch)
    message = SimpleNamespace(
        content="beep boop",
        id=100,
        author=SimpleNamespace(id=55, name="robot", display_name="Robot", bot=True),
        channel=SimpleNamespace(id=123),
        guild=SimpleNamespace(id=7),
    )

    input_id = run(service.handle_discord_message(message))

    assert input_id is None
    assert service._publisher.published == []
    assert service._pending_routes == {}


def test_ranked_response_respects_thread_reply_and_policy_override(monkeypatch):
    service, fake_clock, discord_client = build_service(monkeypatch)
    payload = ResponseRankedPayload(
        final_response="done",
        input_id="in-1",
        channel_id="123",
        thread_id="999",
        reply_to_message_id="777",
        interaction_policy={"delay_seconds": 0.4, "typing_seconds": 0.2, "cooldown_seconds": 0.5},
    )
    msg = DummyMsg(payload.to_json())

    run(service._handle_ranked_response(msg))

    assert msg.acked
    assert not msg.nacked
    assert fake_clock.now == pytest.approx(100.6, abs=0.001)
    assert discord_client.thread.partial_message_requests == [777]
    sent_content, sent_kwargs = discord_client.thread.messages[-1]
    assert sent_content == "done"
    assert sent_kwargs["reference"] == DummyMessageReference(message_id=777, channel_id=999, fail_if_not_exists=False)


def test_ranked_response_applies_cooldown_per_channel(monkeypatch):
    service, fake_clock, _discord_client = build_service(monkeypatch)
    first = DummyMsg(
        ResponseRankedPayload(
            final_response="first",
            channel_id="123",
            interaction_policy={"delay_seconds": 0.0, "typing_seconds": 0.0, "cooldown_seconds": 1.0},
        ).to_json()
    )
    second = DummyMsg(
        ResponseRankedPayload(
            final_response="second",
            channel_id="123",
            interaction_policy={"delay_seconds": 0.0, "typing_seconds": 0.0, "cooldown_seconds": 1.0},
        ).to_json()
    )

    run(service._handle_ranked_response(first))
    before_second = fake_clock.now
    run(service._handle_ranked_response(second))

    assert second.acked
    assert fake_clock.now - before_second >= 1.0


def test_ranked_response_invalid_payload_naks(monkeypatch):
    service, _fake_clock, _discord_client = build_service(monkeypatch)
    msg = DummyMsg(json.dumps(["bad"]))

    run(service._handle_ranked_response(msg))

    assert msg.nacked
    assert msg.nak_calls == 1
    assert msg.ack_calls == 0


def test_send_channel_message_invalid_channel_or_thread_id_returns_true(monkeypatch):
    service, _fake_clock, discord_client = build_service(monkeypatch)
    sent_channel = run(
        service._send_channel_message(
            "not-a-channel-id",
            content="ignored",
            reply_to_message_id=None,
            thread_id=None,
            author_id=None,
            interaction_metadata=None,
        )
    )
    sent_thread = run(
        service._send_channel_message(
            "123",
            content="ignored",
            reply_to_message_id=None,
            thread_id="not-a-thread-id",
            author_id=None,
            interaction_metadata=None,
        )
    )

    assert sent_channel is True
    assert sent_thread is True
    assert discord_client.channel.messages == []
    assert discord_client.thread.messages == []


def test_send_channel_message_falls_back_to_fetch_channel_for_cache_miss(monkeypatch):
    service, _fake_clock, discord_client = build_service(monkeypatch)
    discord_client._cache.pop(123)

    sent = run(
        service._send_channel_message(
            "123",
            content="hello from fetch",
            reply_to_message_id="444",
            thread_id=None,
            author_id=None,
            interaction_metadata=None,
        )
    )

    assert sent is True
    assert discord_client.fetch_calls == [123]
    assert discord_client.channel.partial_message_requests == [444]
    sent_content, sent_kwargs = discord_client.channel.messages[-1]
    assert sent_content == "hello from fetch"
    assert sent_kwargs["reference"] == DummyMessageReference(message_id=444, channel_id=123, fail_if_not_exists=False)


def test_send_channel_message_missing_channel_lookup_returns_true(monkeypatch):
    service, _fake_clock, discord_client = build_service(monkeypatch)
    sent = run(
        service._send_channel_message(
            "321",
            content="ignored",
            reply_to_message_id=None,
            thread_id=None,
            author_id=None,
            interaction_metadata=None,
        )
    )

    assert sent is True
    assert discord_client.channel.messages == []
    assert discord_client.fetch_calls == [321]


def test_send_channel_message_applies_typing_delay_and_cooldown(monkeypatch):
    service, fake_clock, discord_client = build_service(monkeypatch)
    sent = run(
        service._send_channel_message(
            "123",
            content="hello",
            reply_to_message_id="111",
            thread_id=None,
            author_id="42",
            interaction_metadata={"delay_seconds": 0.4, "typing_seconds": 0.2, "cooldown_seconds": 0.5},
        )
    )

    assert sent is True
    assert fake_clock.now == pytest.approx(100.6, abs=0.001)
    assert discord_client.channel.typing_entries == 1
    sent_content, sent_kwargs = discord_client.channel.messages[-1]
    assert sent_content == "hello"
    assert sent_kwargs["reference"] == DummyMessageReference(message_id=111, channel_id=123, fail_if_not_exists=False)
    assert service._cooldown_until["channel:123"] == pytest.approx(101.1, abs=0.001)
    assert service._cooldown_until["user:42"] == pytest.approx(101.1, abs=0.001)


def test_send_channel_message_uses_fetched_thread_for_delivery_and_reply_reference(monkeypatch):
    service, _fake_clock, discord_client = build_service(monkeypatch)
    discord_client._cache.pop(999)

    sent = run(
        service._send_channel_message(
            "123",
            content="thread hello",
            reply_to_message_id="222",
            thread_id="999",
            author_id="42",
            interaction_metadata=None,
        )
    )

    assert sent is True
    assert discord_client.fetch_calls == [999]
    assert discord_client.thread.partial_message_requests == [222]
    sent_content, sent_kwargs = discord_client.thread.messages[-1]
    assert sent_content == "thread hello"
    assert sent_kwargs["reference"] == DummyMessageReference(message_id=222, channel_id=999, fail_if_not_exists=False)


def test_ranked_response_missing_channel_mapping_acks_once(monkeypatch):
    service, _fake_clock, _discord_client = build_service(monkeypatch)
    msg = DummyMsg(ResponseRankedPayload(final_response="done", input_id="unknown").to_json())

    run(service._handle_ranked_response(msg))

    assert msg.acked is True
    assert msg.nacked is False
    assert msg.ack_calls == 1
    assert msg.nak_calls == 0


def test_ranked_response_accepts_enveloped_payload(monkeypatch):
    service, _fake_clock, discord_client = build_service(monkeypatch)
    payload = ResponseRankedPayload(final_response="wrapped", channel_id="123", input_id="in-wrap")
    envelope = EventEnvelope.build(
        subject=EventSubjects.RESPONSE_RANKED,
        payload=json.loads(payload.to_json()),
        producer="selector",
    )
    msg = DummyMsg(json.dumps(envelope.__dict__))

    run(service._handle_ranked_response(msg))

    assert msg.acked
    assert discord_client.channel.messages[-1][0] == "wrapped"


def test_reaction_and_message_edit_emit_feedback_signals(monkeypatch):
    service, _fake_clock, _discord_client = build_service(monkeypatch)
    message = SimpleNamespace(
        content="hello",
        id=199,
        author=SimpleNamespace(id=5, name="alice", display_name="Alice", bot=False),
        channel=SimpleNamespace(id=123),
        guild=SimpleNamespace(id=7),
        attachments=[],
        reference=None,
        thread=None,
    )
    run(service.handle_discord_message(message))

    reaction = SimpleNamespace(message=SimpleNamespace(id=199), emoji="👍")
    user = SimpleNamespace(id=42, bot=False)
    run(service.handle_discord_reaction(reaction, user))

    edited_before = SimpleNamespace(content="hello")
    edited_after = SimpleNamespace(content="hello updated", id=199, author=SimpleNamespace(id=42))
    run(service.handle_discord_message_edit(edited_before, edited_after))

    subjects = [entry[0] for entry in service._publisher.published]
    assert EventSubjects.DISCORD_FEEDBACK_SIGNAL in subjects
    feedback_events = [entry for entry in service._publisher.published if entry[0] == EventSubjects.DISCORD_FEEDBACK_SIGNAL]
    assert len(feedback_events) == 2
    assert feedback_events[0][1]["payload"]["signal_type"] == "reaction"
    assert feedback_events[0][1]["payload"]["input_id"] is not None
    assert feedback_events[1][1]["payload"]["signal_type"] == "message_edit"


def test_ranked_response_egress_policy_escalates_ambiguous_confidence(monkeypatch):
    service, _fake_clock, discord_client = build_service(monkeypatch)
    payload = ResponseRankedPayload(
        final_response="I can help you bypass password controls.",
        input_id="in-esc",
        channel_id="123",
        confidence=0.7,
        candidates=[
            {
                "text": "I can help you bypass password controls.",
                "confidence": 0.7,
                "source": "responder:persona",
                "safety_metadata": {
                    "policy_artifacts": [
                        {"stage": "pre_generation", "risk_level": "ambiguous", "policy_version": "v1"}
                    ]
                },
            }
        ],
    )
    msg = DummyMsg(payload.to_json())

    run(service._handle_ranked_response(msg))

    assert msg.acked
    assert discord_client.channel.messages == []
    telemetry = [item for item in service._publisher.published if item[0] == "dtr.telemetry.egress_policy.v1"]
    assert telemetry
    assert telemetry[0][1]["payload"]["decision_action"] == "escalate"
