import asyncio
import json
import os
import subprocess
import uuid
from pathlib import Path

import nats
import pytest
from nats.js.api import DiscardPolicy, RetentionPolicy, StorageType, StreamConfig

from deepthought.modules import BasicMemory, InputHandler, LLMStub, OutputHandler
from tests.helpers import nats_server_available


def _write_trace(path: Path, actions) -> None:
    with path.open("w", encoding="utf-8") as f:
        for action in actions:
            json.dump(
                {
                    "state": "s",
                    "action": action,
                    "reward": 0.0,
                    "latency": 0.1,
                    "timestamp": "2024-01-01T00:00:00",
                },
                f,
            )
            f.write("\n")


def test_replay_cli(tmp_path: Path) -> None:
    trial = tmp_path / "trial.json"
    golden = tmp_path / "golden.json"
    _write_trace(trial, ["hello world"])
    _write_trace(golden, ["hello world"])

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(Path(__file__).resolve().parents[1] / "src"),
            str(Path(__file__).resolve().parent),
        ]
    )
    result = subprocess.run(
        [
            "python",
            str(Path(__file__).resolve().parents[1] / "tools" / "replay.py"),
            str(trial),
            str(golden),
        ],
        stdout=subprocess.PIPE,
        text=True,
        check=True,
        env=env,
    )
    out = result.stdout
    assert "bleu:" in out
    assert "rouge_l:" in out
    assert "avg_latency:" in out
    assert "actions_per_second:" in out


def test_replay_cli_mismatch(tmp_path: Path) -> None:
    trial = tmp_path / "trial_m.json"
    golden = tmp_path / "golden_m.json"
    _write_trace(trial, ["foo"])
    _write_trace(golden, ["bar"])

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(Path(__file__).resolve().parents[1] / "src"),
            str(Path(__file__).resolve().parent),
        ]
    )
    result = subprocess.run(
        [
            "python",
            str(Path(__file__).resolve().parents[1] / "tools" / "replay.py"),
            str(trial),
            str(golden),
        ],
        stdout=subprocess.PIPE,
        text=True,
        check=True,
        env=env,
    )
    out = result.stdout
    assert "bleu:" in out
    assert "rouge_l:" in out
    assert "avg_latency:" in out
    assert "actions_per_second:" in out


@pytest.mark.asyncio
@pytest.mark.nats
async def test_multi_agent_round_robin(tmp_path: Path) -> None:
    """Agents should exchange at least one message using NATS."""
    if not nats_server_available():
        pytest.skip("NATS server not available")

    nc = await nats.connect("nats://localhost:4222", name="pytest_multi_agent")
    js = nc.jetstream()

    try:
        await js.stream_info("deepthought_events")
    except Exception:
        cfg = StreamConfig(
            name="deepthought_events",
            subjects=["dtr.>"],
            retention=RetentionPolicy.LIMITS,
            storage=StorageType.MEMORY,
            max_msgs_per_subject=100,
            discard=DiscardPolicy.OLD,
        )
        await js.add_stream(cfg)

    memory_file = tmp_path / "mem.json"
    memory = BasicMemory(nc, js, memory_file=str(memory_file))
    llm = LLMStub(nc, js)

    input_handlers = [InputHandler(nc, js) for _ in range(3)]
    received: list[str] = []
    done = asyncio.Event()

    def make_callback(idx: int):
        def cb(input_id: str, text: str) -> None:
            received.append(text)
            next_idx = (idx + 1) % 3
            if len(received) >= 3:
                done.set()
            else:
                asyncio.create_task(input_handlers[next_idx].process_input(text))

        return cb

    output_handlers = [OutputHandler(nc, js, output_callback=make_callback(i)) for i in range(3)]

    results = await asyncio.gather(
        memory.start_listening(durable_name=f"mem_{uuid.uuid4()}"),
        llm.start_listening(durable_name=f"llm_{uuid.uuid4()}"),
        *(oh.start_listening(durable_name=f"out_{i}_{uuid.uuid4()}") for i, oh in enumerate(output_handlers)),
        return_exceptions=True,
    )
    for result in results:
        if result is False or isinstance(result, Exception):
            pytest.fail(f"Listener failed: {result}")

    await asyncio.sleep(0.5)
    await input_handlers[0].process_input("hello")

    await asyncio.wait_for(done.wait(), timeout=10.0)

    assert len(received) >= 3

    await memory.stop_listening()
    await llm.stop_listening()
    await asyncio.gather(*(oh.stop_listening() for oh in output_handlers))
    await nc.drain()
