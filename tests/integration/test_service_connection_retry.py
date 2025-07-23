import pytest

pytest.importorskip("nats")

from deepthought.services.code_generation_service import CodeGenerationService


@pytest.mark.asyncio
async def test_service_start_fails_without_nats(monkeypatch, unused_tcp_port):
    url = f"nats://localhost:{unused_tcp_port}"
    service = CodeGenerationService(
        nats_url=url, connect_retries=1, connect_timeout=1
    )

    async def fail_connect(*args, **kwargs):
        raise OSError("fail")

    monkeypatch.setattr(service._nc, "connect", fail_connect)

    started = await service.start()
    assert started is False
    await service.stop()
