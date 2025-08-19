"""Configuration for the perception service."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class PerceptionConfig(BaseSettings):
    """Settings controlling the perception service."""

    nats_url: str = Field("nats://localhost:4222", env="DT_NATS_URL")
    text_model: str = "intfloat/e5-small-v2"
    text_hop_size: float = 0.03
    text_cache_dir: str | None = None

    audio_model: str = "wavlm"
    audio_model_path: str | None = None
    audio_window_size: float = 0.02
    audio_hop_size: float = 0.01
    audio_cache_dir: str | None = None

    video_model: str = "siglip"
    video_hop_size: float = 1.0
    video_cache_dir: str | None = None

    wandb_project: str | None = Field(None, env="DT_WANDB_PROJECT")
    wandb_sweep_id: str | None = Field(None, env="DT_WANDB_SWEEP_ID")

    class Config:
        env_prefix = "DT_PERCEPTION_"
