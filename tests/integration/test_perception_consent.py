from unittest.mock import AsyncMock

import pytest

from deepthought.services.perception.service import PerceptionService


class DummyPublisher:
    def __init__(self):
        self.publish = AsyncMock()


@pytest.mark.asyncio
async def test_audio_consent_required(monkeypatch):
    monkeypatch.setenv("DT_REQUIRE_AUDIO_CONSENT", "1")
    monkeypatch.delenv("DT_AUDIO_CONSENT", raising=False)

    service = PerceptionService(publisher=DummyPublisher(), audio_worker=object())

    with pytest.raises(PermissionError):
        await service.run("m1", "u1", audio_path="dummy.wav")

    service.publisher.publish.assert_not_awaited()
