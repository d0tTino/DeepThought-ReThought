"""Unit tests for service layer using a mocked bus."""

from __future__ import annotations

import json

import pytest

from src.deepthought.bus import Publisher, Subscriber
from src.deepthought.eda.events import EventSubjects
from src.deepthought.services import (
    SocialGraphService,
    QuestLogService,
    PerceptionService,
    ResponderService,
    SelectorService,
)


class FakeMsg:
    def __init__(self, payload):
        self.data = json.dumps(payload).encode("utf-8")
        self.acked = False

    async def ack(self):
        self.acked = True


class MockPublisher(Publisher):
    def __init__(self):
        self.published = []

    async def publish(self, subject: str, payload):
        self.published.append((subject, payload))


class MockSubscriber(Subscriber):
    def __init__(self):
        self.handlers = {}

    async def subscribe(self, subject, handler, queue="", use_jetstream=False, durable=""):
        self.handlers[(subject, durable)] = {
            "handler": handler,
            "use_jetstream": use_jetstream,
        }


@pytest.mark.asyncio
async def test_social_graph_service_ack_and_publish():
    publisher = MockPublisher()
    subscriber = MockSubscriber()
    service = SocialGraphService(subscriber, publisher)

    await service.start()
    key = (EventSubjects.SOCIAL_GRAPH_UPDATE, SocialGraphService.DURABLE_NAME)
    handler = subscriber.handlers[key]["handler"]
    assert subscriber.handlers[key]["use_jetstream"] is True

    msg = FakeMsg(
        {
            "user_id": "u1",
            "updates": {"edges": [{"target": "u2", "weight": 0.9}]},
        }
    )
    await handler(msg)
    assert msg.acked is True
    assert publisher.published[0][0] == EventSubjects.SOCIAL_GRAPH_SNAPSHOT
    snapshot_payload = publisher.published[0][1]
    assert snapshot_payload["graph"]["edges"]["u2"] == 0.9


@pytest.mark.asyncio
async def test_questlog_service_publishes_snapshot_and_done():
    publisher = MockPublisher()
    subscriber = MockSubscriber()
    service = QuestLogService(subscriber, publisher)

    await service.start()

    create_handler = subscriber.handlers[(EventSubjects.QUEST_CREATE, QuestLogService.CREATE_DURABLE)]["handler"]
    update_handler = subscriber.handlers[(EventSubjects.QUEST_UPDATE, QuestLogService.UPDATE_DURABLE)]["handler"]

    create_msg = FakeMsg({"quest_id": "q1", "name": "Intro"})
    await create_handler(create_msg)
    assert create_msg.acked is True
    assert publisher.published[0][0] == EventSubjects.QUEST_SNAPSHOT

    update_msg = FakeMsg({"quest_id": "q1", "status": "completed", "progress": 1.0})
    await update_handler(update_msg)
    assert update_msg.acked is True
    subjects = [subject for subject, _ in publisher.published]
    assert EventSubjects.QUEST_DONE in subjects


@pytest.mark.asyncio
async def test_perception_service_fuses_modalities():
    publisher = MockPublisher()
    subscriber = MockSubscriber()
    service = PerceptionService(subscriber, publisher)

    await service.start()

    audio_handler = subscriber.handlers[(EventSubjects.PERCEPTION_AUDIO_EMBED, PerceptionService.AUDIO_DURABLE)]["handler"]
    image_handler = subscriber.handlers[(EventSubjects.PERCEPTION_IMAGE_EMBED, PerceptionService.IMAGE_DURABLE)]["handler"]

    audio_msg = FakeMsg({"input_id": "in1", "embedding": [0.1], "metadata": {"energy": 0.5}})
    image_msg = FakeMsg({"input_id": "in1", "embedding": [0.2], "metadata": {"brightness": 0.8}})

    await audio_handler(audio_msg)
    await image_handler(image_msg)

    assert audio_msg.acked is True
    assert image_msg.acked is True
    assert publisher.published[-1][0] == EventSubjects.PERCEPTION_FUSED
    fused_payload = publisher.published[-1][1]
    assert fused_payload["features"]["audio"] == [0.1]
    assert fused_payload["features"]["image"] == [0.2]


@pytest.mark.asyncio
async def test_responder_service_creates_candidates():
    publisher = MockPublisher()
    subscriber = MockSubscriber()
    service = ResponderService(subscriber, publisher)

    await service.start()

    memory_handler = subscriber.handlers[(EventSubjects.MEMORY_RETRIEVED, ResponderService.MEMORY_DURABLE)]["handler"]
    fused_handler = subscriber.handlers[(EventSubjects.PERCEPTION_FUSED, ResponderService.FUSED_DURABLE)]["handler"]

    memory_msg = FakeMsg({"input_id": "in1", "retrieved_knowledge": {"summary": "Hello"}})
    fused_msg = FakeMsg({"input_id": "in1", "features": {"audio": [0.1], "image": [0.2]}})

    await memory_handler(memory_msg)
    await fused_handler(fused_msg)

    assert memory_msg.acked is True
    assert fused_msg.acked is True
    assert publisher.published[-1][0] == EventSubjects.RESPONSE_CANDIDATES
    candidates = publisher.published[-1][1]["candidates"]
    assert candidates[0]["confidence"] >= candidates[1]["confidence"]


@pytest.mark.asyncio
async def test_selector_service_ranks_candidates():
    publisher = MockPublisher()
    subscriber = MockSubscriber()
    service = SelectorService(subscriber, publisher)

    await service.start()
    handler = subscriber.handlers[(EventSubjects.RESPONSE_CANDIDATES, SelectorService.CANDIDATE_DURABLE)]["handler"]

    msg = FakeMsg(
        {
            "input_id": "in1",
            "candidates": [
                {"id": "a", "confidence": 0.2},
                {"id": "b", "confidence": 0.9},
            ],
        }
    )
    await handler(msg)

    assert msg.acked is True
    assert publisher.published[-1][0] == EventSubjects.RESPONSE_RANKED
    ranked_payload = publisher.published[-1][1]
    assert ranked_payload["ranked_candidates"][0]["id"] == "b"
    assert ranked_payload["selected_index"] == 0
