import json
import os
import subprocess
from pathlib import Path


def _write_trace(path: Path, responses):
    with path.open("w", encoding="utf-8") as f:
        for resp in responses:
            obj = {
                "event": "RESPONSE_GENERATED",
                "payload": {"final_response": resp, "timestamp": "2024-01-01T00:00:00"},
            }
            f.write(json.dumps(obj) + "\n")


def test_replay_cli(tmp_path: Path) -> None:
    trial = tmp_path / "trial.jsonl"
    golden = tmp_path / "golden.jsonl"
    _write_trace(trial, ["hello world"])
    _write_trace(golden, ["hello world"])

    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    result = subprocess.run(
        [
            "python",
            str(Path(__file__).resolve().parents[1] / "tools" / "replay.py"),
            str(trial),
            str(golden),
            "--nats",
            "",
        ],
        stdout=subprocess.PIPE,
        text=True,
        check=True,
        env=env,
    )
    out = result.stdout
    assert "bleu:" in out
    assert "rouge_l:" in out
