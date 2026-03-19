import pytest

pytest.importorskip("aiosqlite")

from deepthought.services.db_manager import DBManager


@pytest.mark.asyncio
async def test_store_and_recall_multimodal_memory(tmp_path):
    db = DBManager(str(tmp_path / "db.sqlite"))
    payload = {
        "schema_version": "multimodal.memory.v1",
        "input_id": "input-1",
        "user_id": "user-1",
        "channel_id": "channel-1",
        "observed_at": "2026-03-19T00:00:00+00:00",
        "summary": "attachments[image:1]",
        "attachment_summary": "image:1",
        "attachments": [
            {
                "attachment_index": 0,
                "media_type": "image",
                "content_type": "image/png",
                "filename": "pet.png",
                "url": "https://example.test/pet.png",
                "summary": "image attachment present",
                "confidence": 0.7,
            }
        ],
        "confidence": [
            {
                "label": "aggregate",
                "value": 0.7,
                "uncertain": False,
                "reason": "",
            }
        ],
        "entities": [],
        "observations": [
            {
                "observation_id": "input-1:image:0",
                "observation_type": "event_summary",
                "modality": "image",
                "summary": "user posted image attachment",
                "confidence": 0.7,
                "uncertain": False,
                "happened_at": "2026-03-19T00:00:00+00:00",
                "input_id": "input-1",
                "user_id": "user-1",
                "channel_id": "channel-1",
            }
        ],
    }

    await db.store_multimodal_memory(payload)
    rows = await db.recall_multimodal_memories("user-1", channel_id="channel-1", limit=2)

    assert rows[0]["input_id"] == "input-1"
    assert rows[0]["attachments"][0]["media_type"] == "image"
    assert rows[0]["confidence"][0]["label"] == "aggregate"
    assert rows[0]["observations"][0]["summary"] == "user posted image attachment"

    facts = await db.recall_user("user-1", limit=5)
    assert any(memory == "attachments[image:1]" for _topic, memory in facts)

    await db.close()
