import logging

import pytest

pytest.importorskip("nats")
import nats

from deepthought.eda.publisher import Publisher


class DummyNATS:
    def __init__(self):
        self.is_connected = True


class TimeoutJS:
    async def publish(self, subject, data, timeout=10.0):
        raise nats.errors.TimeoutError("too slow")


@pytest.mark.asyncio
async def test_publish_timeout_message(caplog):
    nc = DummyNATS()
    js = TimeoutJS()
    pub = Publisher(nc, js)

    with caplog.at_level(logging.ERROR):
        with pytest.raises(nats.errors.TimeoutError) as exc:
            await pub.publish("topic", {"foo": "bar"})

    msg = exc.value.args[0]
    assert "Publish timeout" in msg
    assert "topic" in msg
    assert "foo" in msg
    assert any("Publish timeout" in r.getMessage() for r in caplog.records)
