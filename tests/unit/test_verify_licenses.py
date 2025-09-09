from __future__ import annotations

import json
from pathlib import Path
import importlib.util

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "verify_licenses.py"
spec = importlib.util.spec_from_file_location("verify_licenses", SCRIPT_PATH)
verify_licenses = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify_licenses)


def _write_whitelist(path: Path) -> Path:
    data = {
        "text_model": "intfloat/e5-small-v2@9d1db5bedc62f5d6c594bb4a7f14c04b1e5e6a0c",
        "audio_model": "wavlm@60e4d1c438ae0de1f5c9463c5b44e7c2f7b2a7fa",
        "video_model": "siglip@4a5b96c2d9b60f1a8327db6a36e5b92a9c3ad6fa",
        "licenses": {
            "text_model": {"component": "E5", "license": "MIT"},
            "audio_model": {"component": "WavLM", "license": "MIT"},
            "video_model": {"component": "SigLIP", "license": "Apache-2.0"},
        },
    }
    whitelist = path / "model_version_whitelist.json"
    whitelist.write_text(json.dumps(data))
    return whitelist


def _write_licenses(path: Path, entries: list[tuple[str, str]]) -> Path:
    lines = ["| Component | License |", "| --- | --- |"]
    lines.extend([f"| {name} | {license} |" for name, license in entries])
    license_file = path / "licenses.md"
    license_file.write_text("\n".join(lines))
    return license_file


def test_verify_licenses_pass(monkeypatch, tmp_path):
    whitelist = _write_whitelist(tmp_path)
    license_file = _write_licenses(
        tmp_path, [("E5", "MIT"), ("WavLM", "MIT"), ("SigLIP", "Apache-2.0")]
    )
    monkeypatch.setattr(verify_licenses, "LICENSE_FILE", license_file)
    monkeypatch.setattr(verify_licenses, "WHITELIST_PATH", whitelist)
    assert verify_licenses.main() == 0


def test_verify_licenses_missing(monkeypatch, tmp_path):
    whitelist = _write_whitelist(tmp_path)
    license_file = _write_licenses(tmp_path, [("E5", "MIT"), ("WavLM", "MIT")])
    monkeypatch.setattr(verify_licenses, "LICENSE_FILE", license_file)
    monkeypatch.setattr(verify_licenses, "WHITELIST_PATH", whitelist)
    assert verify_licenses.main() == 1
