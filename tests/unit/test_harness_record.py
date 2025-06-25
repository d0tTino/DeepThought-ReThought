from deepthought.harness.record import TraceEvent, record_event


def test_record_event():
    trace = []
    record_event(trace, "state", "action", 0.5, 0.1)
    assert len(trace) == 1
    evt = trace[0]
    assert evt.state == "state"
    assert evt.action == "action"
    assert evt.reward == 0.5
    assert evt.latency == 0.1
