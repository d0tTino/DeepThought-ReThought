"""Automatic speech recognition utilities for the perception service.

This module provides a thin wrapper around the MIT-licensed `openai/whisper`
model family. It exposes :func:`transcribe_audio_tokens` which yields
time-aligned ``(token, start, end)`` tuples for a given audio file and stores
results next to cached audio feature memmaps for later reuse.
"""

from __future__ import annotations

import importlib
import json
import logging
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Tuple

from .config import PerceptionConfig

# A recognised token represented as ``(text, start_time, end_time)`` in seconds.
Token = Tuple[str, float, float]

_LOG = logging.getLogger(__name__)
_CACHE_VERSION = 1
_MODEL_CACHE: dict[tuple[str, str | None], Any] = {}


def _resolve_cache_dir(audio_path: Path, cache_dir: str | Path | None, cfg: PerceptionConfig) -> Path:
    """Return the directory that should contain cache artefacts."""

    if cache_dir is not None:
        base = Path(cache_dir)
    elif cfg.audio_cache_dir is not None:
        base = Path(cfg.audio_cache_dir)
    else:
        base = audio_path.parent
    base.mkdir(parents=True, exist_ok=True)
    return base


def _transcript_cache_path(
    audio_path: Path,
    cache_dir: Path,
    audio_model: str,
    window_size: float,
    step_size: float,
) -> Path:
    """Return the JSON cache path mirroring the audio memmap naming scheme."""

    memmap_name = f"{audio_path.stem}_{audio_model}_ws{window_size}_ss{step_size}.dat"
    memmap_path = cache_dir / memmap_name
    return memmap_path.with_suffix(".transcript.json")


def _load_cached_tokens(
    cache_path: Path,
    *,
    expected_model: str,
    expected_language: str | None,
    expected_mtime_ns: int,
) -> List[Token] | None:
    """Load cached tokens if the metadata matches the current request."""

    try:
        with cache_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except FileNotFoundError:
        return None
    except Exception:  # pragma: no cover - cache corruption is rare
        _LOG.warning("Failed to read ASR cache at %%s", cache_path, exc_info=True)
        return None

    if payload.get("version") != _CACHE_VERSION:
        return None
    if payload.get("model") != expected_model:
        return None
    if payload.get("requested_language") != expected_language:
        return None
    if payload.get("audio_mtime_ns") != expected_mtime_ns:
        return None

    tokens_payload = payload.get("tokens")
    if not isinstance(tokens_payload, list):
        return None

    tokens: List[Token] = []
    try:
        for item in tokens_payload:
            text = str(item["text"])
            start = float(item["start"])
            end = float(item["end"])
            tokens.append((text, start, end))
    except (KeyError, TypeError, ValueError):
        return None

    return tokens


def _store_cached_tokens(
    cache_path: Path,
    tokens: Sequence[Token],
    *,
    model: str,
    language: str | None,
    detected_language: str | None,
    audio_mtime_ns: int,
) -> None:
    """Persist a transcript cache entry in JSON format."""

    payload = {
        "version": _CACHE_VERSION,
        "model": model,
        "requested_language": language,
        "detected_language": detected_language,
        "audio_mtime_ns": audio_mtime_ns,
        "tokens": [
            {"text": text, "start": start, "end": end}
            for text, start, end in tokens
        ],
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    tmp_path.replace(cache_path)


def _load_whisper_model(model_name: str, device: str | None) -> Any:
    """Return a cached Whisper model instance."""

    cache_key = (model_name, device)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    try:
        whisper = importlib.import_module("whisper")
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "The 'whisper' package is required for ASR support. Install it via "
            "'pip install openai-whisper'."
        ) from exc

    load_kwargs: dict[str, Any] = {}
    if device is not None:
        load_kwargs["device"] = device

    model = whisper.load_model(model_name, **load_kwargs)
    _MODEL_CACHE[cache_key] = model
    return model


def _extract_tokens(result: dict[str, Any]) -> List[Token]:
    """Convert a Whisper transcription result into ``Token`` objects."""

    segments: Iterable[dict[str, Any]] = result.get("segments", []) or []
    tokens: List[Token] = []

    for segment in segments:
        words = segment.get("words")
        if words:
            for word in words:
                start = word.get("start")
                end = word.get("end")
                text = word.get("word", "")
                if start is None or end is None:
                    continue
                text = str(text).strip()
                if not text:
                    continue
                tokens.append((text, float(start), float(end)))
        else:
            start = segment.get("start")
            end = segment.get("end")
            text = segment.get("text", "")
            if start is None or end is None:
                continue
            text = str(text).strip()
            if not text:
                continue
            tokens.append((text, float(start), float(end)))

    tokens.sort(key=lambda item: item[1])
    return tokens


def transcribe_audio_tokens(
    audio_path: str | Path,
    *,
    model_name: str = "small",
    language: str | None = None,
    device: str | None = None,
    cache_dir: str | Path | None = None,
    audio_model: str | None = None,
    window_size: float | None = None,
    step_size: float | None = None,
) -> List[Token]:
    """Return a list of ``(token, start, end)`` tuples for ``audio_path``.

    Parameters
    ----------
    audio_path:
        Path to an input audio file compatible with Whisper.
    model_name:
        Whisper model variant to load (e.g. ``"small"`` or ``"base"``).
    language:
        Optional language code. When ``None`` Whisper performs language
        detection automatically.
    device:
        Optional device specifier (e.g. ``"cuda"``) forwarded to Whisper.
    cache_dir:
        Directory in which to store transcript caches. When omitted, the
        perception configuration determines the location and falls back to the
        audio file's parent directory.
    audio_model, window_size, step_size:
        Parameters mirroring the audio feature extraction configuration. They
        ensure transcript caches live beside the corresponding audio memmaps.
    """

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    cfg = PerceptionConfig()
    effective_audio_model = audio_model or cfg.audio_model
    effective_window_size = window_size if window_size is not None else cfg.audio_window_size
    effective_step_size = step_size if step_size is not None else cfg.audio_hop_size

    cache_base = _resolve_cache_dir(audio_path, cache_dir, cfg)
    transcript_path = _transcript_cache_path(
        audio_path,
        cache_base,
        effective_audio_model,
        effective_window_size,
        effective_step_size,
    )

    audio_mtime_ns = audio_path.stat().st_mtime_ns
    cached = _load_cached_tokens(
        transcript_path,
        expected_model=model_name,
        expected_language=language,
        expected_mtime_ns=audio_mtime_ns,
    )
    if cached is not None:
        return cached

    model = _load_whisper_model(model_name, device)
    result = model.transcribe(
        str(audio_path),
        language=language,
        task="transcribe",
        word_timestamps=True,
        verbose=False,
    )

    tokens = _extract_tokens(result)
    detected_language = result.get("language") if isinstance(result, dict) else None

    _store_cached_tokens(
        transcript_path,
        tokens,
        model=model_name,
        language=language,
        detected_language=detected_language,
        audio_mtime_ns=audio_mtime_ns,
    )

    return tokens
