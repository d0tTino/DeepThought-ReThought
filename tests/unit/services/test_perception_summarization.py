from deepthought.services.perception.summarization import build_semantic_notes


def test_build_semantic_notes_generates_stable_schema_with_modalities():
    payload = {
        "modality_confidence": {"image": 0.8, "audio": 0.5},
        "by_modality": {
            "image": {"spans": [[0, 1]], "embeddings": [[0.1, 0.2]]},
            "audio": {"spans": [[0, 10], [11, 20]], "embeddings": [[0.1], [0.2]]},
        },
    }

    result = build_semantic_notes(
        attachments=[{"content_type": "image/png"}, {"content_type": "audio/wav"}],
        embeddings_payload=payload,
    )

    assert result["schema_version"] == "multimodal.semantic-notes.v1"
    assert len(result["notes"]) == 2
    assert result["by_modality"]["image"]["what"].startswith("1 embedding vectors")
    assert "attachments[audio:1, image:1]" == result["attachments"]


def test_build_semantic_notes_marks_low_confidence_for_fallback():
    result = build_semantic_notes(
        attachments=[{"content_type": "image/png"}],
        embeddings_payload={
            "modality_confidence": {"image": 0.2},
            "by_modality": {"image": {"spans": [], "embeddings": []}},
        },
    )

    assert result["confidence"]["low_confidence"] is True
    assert result["fallback"]["ask_clarifying_question"] is True
