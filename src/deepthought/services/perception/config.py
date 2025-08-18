"""Configuration for the perception service."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class PerceptionConfig(BaseSettings):
    """Settings controlling the perception service."""

    nats_url: str = Field("nats://localhost:4222", env="DT_NATS_URL")
    audio_model: str = "wavlm"
    audio_model_path: str | None = None
    audio_window_size: float = 0.02
    audio_step_size: float = 0.01
    audio_cache_dir: str | None = None

    class Config:
        env_prefix = "DT_PERCEPTION_"
