"""Configuration for the perception service."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class PerceptionConfig(BaseSettings):
    """Settings controlling the perception service."""

    nats_url: str = Field("nats://localhost:4222", env="DT_NATS_URL")
    # Default text embedding model (E5). Alternative: BGE.
    # See docs/licenses.md for license details.
    text_model: str = "intfloat/e5-small-v2@9d1db5bedc62f5d6c594bb4a7f14c04b1e5e6a0c"
    text_hop_size: float = 0.03
    text_cache_dir: str | None = None

    audio_model: str = "wavlm@60e4d1c438ae0de1f5c9463c5b44e7c2f7b2a7fa"
    audio_model_path: str | None = None
    audio_window_size: float = 0.02
    audio_hop_size: float = 0.01
    audio_cache_dir: str | None = None

    video_model: str = "siglip@4a5b96c2d9b60f1a8327db6a36e5b92a9c3ad6fa"
    video_hop_size: float = 1.0
    video_cache_dir: str | None = None

    wandb_project: str | None = Field(None, env="DT_WANDB_PROJECT")
    wandb_sweep_id: str | None = Field(None, env="DT_WANDB_SWEEP_ID")

    class Config:
        env_prefix = "DT_PERCEPTION_"
