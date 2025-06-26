from datetime import datetime

import deepthought.harness.record as record
from deepthought.harness.record import TraceEvent, record_event


def test_record_event():
    trace = []
    times = [
        datetime(2021, 1, 1, 0, 0, 0),
        datetime(2021, 1, 1, 0, 0, 1),
    ]

    class DummyDT:
        def utcnow(self):
            return times.pop(0)

    original_dt = record.datetime
    record.datetime = DummyDT()  # type: ignore

    try:
        record_event(trace, "state", "action", 0.5, 0.1)
        assert len(trace) == 1
        evt = trace[0]
        assert evt.state == "state"
        assert evt.action == "action"
        assert evt.reward == 0.5
        assert evt.latency == 0.1
        assert evt.timestamp_delta == 0.0

        record_event(trace, "state2", "action2", 1.0, 0.2)
        evt2 = trace[1]
        assert evt2.timestamp_delta == 1.0
    finally:
        record.datetime = original_dt
