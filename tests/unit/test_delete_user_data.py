import importlib
import json
from pathlib import Path


def test_delete_user_data_clears_files_and_embeddings(tmp_path, monkeypatch):
    user_id = "u123"
    text_dir = tmp_path / "text"
    audio_dir = tmp_path / "audio"
    video_dir = tmp_path / "video"
    for d in (text_dir, audio_dir, video_dir):
        d.mkdir()

    (text_dir / f"{user_id}_feat.dat").write_text("x")
    (audio_dir / f"a_{user_id}.dat").write_text("x")
    (video_dir / f"{user_id}.npy").write_text("x")

    emb_path = tmp_path / "emb.json"
    emb_path.write_text(json.dumps({user_id: [1.0], "other": [2.0]}))

    monkeypatch.setenv("DT_PERCEPTION_TEXT_CACHE_DIR", str(text_dir))
    monkeypatch.setenv("DT_PERCEPTION_AUDIO_CACHE_DIR", str(audio_dir))
    monkeypatch.setenv("DT_PERCEPTION_VIDEO_CACHE_DIR", str(video_dir))
    monkeypatch.setenv("DT_USER_EMBEDDINGS_PATH", str(emb_path))

    mod = importlib.import_module("deepthought.services.perception.delete_user_data")

    async def fake_trigger(*_, **__):
        pass

    monkeypatch.setattr(mod, "trigger_replay_jobs", fake_trigger)
    mod.delete_user_data(user_id, nats_url="nats://example")

    data = json.loads(emb_path.read_text())
    assert user_id not in data
    assert "other" in data
