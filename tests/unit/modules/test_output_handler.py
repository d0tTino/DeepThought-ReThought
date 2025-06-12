import pytest

import deepthought.modules.output_handler as output_handler
from deepthought.eda.events import ResponseGeneratedPayload


class DummyNATS:
    def __init__(self):
        self.is_connected = True


class DummyJS:
    pass


class DummySubscriber:
    def __init__(self, *args, **kwargs):
        pass

    async def subscribe(self, *args, **kwargs):
        pass

    async def unsubscribe_all(self):
        pass


class DummyMsg:
    def __init__(self, data):
        self.data = data.encode()
        self.acked = False

    async def ack(self):
        self.acked = True


def create_handler(monkeypatch, callback=None):
    monkeypatch.setattr(output_handler, "Subscriber", DummySubscriber)
    return output_handler.OutputHandler(DummyNATS(), DummyJS(), callback)


@pytest.mark.asyncio
async def test_handle_response_success(monkeypatch):
    received = {}

    def cb(iid, resp):
        received["id"] = iid
        received["resp"] = resp

    handler = create_handler(monkeypatch, cb)
    payload = ResponseGeneratedPayload(final_response="ok", input_id="42")
    msg = DummyMsg(payload.to_json())
    await handler._handle_response_event(msg)

    assert handler.get_response("42") == "ok"
    assert received["id"] == "42"
    assert received["resp"] == "ok"
    assert msg.acked


@pytest.mark.asyncio
async def test_handle_response_error(monkeypatch):
    handler = create_handler(monkeypatch)
    msg = DummyMsg("not json")
    await handler._handle_response_event(msg)

    assert handler.get_all_responses() == {}
    assert not msg.acked
