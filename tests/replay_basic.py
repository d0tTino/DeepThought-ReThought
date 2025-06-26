import json
import os
import subprocess
from pathlib import Path


def _write_trace(path: Path, actions):
    data = []
    for action in actions:
        data.append(
            {
                "state": "s",
                "action": action,
                "reward": 0.0,
                "latency": 0.1,
                "timestamp": "2024-01-01T00:00:00",
            }
        )
    path.write_text(json.dumps(data))


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
