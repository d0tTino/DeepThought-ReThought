import pytest

pytest.importorskip("nats")

from setup_jetstream import JetStreamSetupError, check_nats_server_running, setup_jetstream


@pytest.mark.asyncio
async def test_setup_jetstream_raises_without_server():
    """setup_jetstream should raise when no NATS server is available."""
    if check_nats_server_running():
        pytest.skip("NATS server running; cannot test failure path")

    with pytest.raises(JetStreamSetupError):
        await setup_jetstream()
