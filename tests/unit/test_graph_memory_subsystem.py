import sqlite3

from deepthought.memory.fact_extractor import extract_typed_fact_triples_from_turn
from deepthought.memory.graph import (
    InMemoryGraphMemoryStore,
    ingest_conversation_turns,
    migrate_sqlite_memories_to_graph,
    retrieve_topic_context,
    retrieve_user_context,
)
from deepthought.memory.graph.pipeline import ingest_fact_triples


def test_extract_typed_fact_triples_from_turn():
    triples = extract_typed_fact_triples_from_turn(
        user_id="u1",
        message="Call me Ace. I enjoy hiking and chess. My favorite color is blue. Tomorrow I will practice openings.",
        timestamp="2024-01-01T00:00:00+00:00",
        source_id="turn-1",
    )

    predicates = {t["predicate"] for t in triples}
    assert "has_nickname" in predicates
    assert "likes_hobby" in predicates
    assert "favorite" in predicates
    assert "plans_event" in predicates


def test_ingest_and_retrieve_dedup_ranked():
    store = InMemoryGraphMemoryStore()
    ingest_conversation_turns(
        [
            {"user_id": "u1", "text": "I like chess", "timestamp": "2024-01-01T00:00:00+00:00", "input_id": "a"},
            {
                "user_id": "u1",
                "text": "My favorite game is chess",
                "timestamp": "2024-01-02T00:00:00+00:00",
                "input_id": "b",
            },
        ],
        store,
    )

    user_ctx = retrieve_user_context(store, "u1", limit=10)
    topic_ctx = retrieve_topic_context(store, "chess", limit=10)

    assert user_ctx
    assert topic_ctx
    assert len({e.summary for e in topic_ctx}) == len(topic_ctx)
    assert topic_ctx[0].score >= topic_ctx[-1].score


def test_ingest_assigns_typed_relations_for_fact_types():
    store = InMemoryGraphMemoryStore()
    ingest_conversation_turns(
        [
            {
                "user_id": "u1",
                "text": "My favorite drink is tea. Tomorrow I will run a marathon.",
                "timestamp": "2024-01-03T00:00:00+00:00",
                "input_id": "turn-typed",
            }
        ],
        store,
    )

    relation_types = {rel.relation_type for rel in store.relations.values()}
    assert "preference" in relation_types
    assert "temporal_fact" in relation_types


def test_topic_retrieval_prioritizes_user_scoped_high_confidence_facts():
    store = InMemoryGraphMemoryStore()
    ingest_conversation_turns(
        [
            {"user_id": "u1", "text": "My favorite game is chess", "timestamp": "2024-01-04T00:00:00+00:00", "input_id": "h1"},
            {"user_id": "u2", "text": "chess", "timestamp": "2024-01-01T00:00:00+00:00", "input_id": "l1"},
        ],
        store,
    )

    topic_ctx = retrieve_topic_context(store, "chess", limit=5)

    assert topic_ctx
    assert topic_ctx[0].attributes.get("user_scoped") is True
    assert topic_ctx[0].confidence >= topic_ctx[-1].confidence


def test_migrate_sqlite_memories_to_graph(tmp_path):
    db_path = tmp_path / "migrate.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE memories (user_id TEXT, topic TEXT, memory TEXT, sentiment_score REAL, timestamp DATETIME)"
    )
    conn.execute(
        "INSERT INTO memories (user_id, topic, memory, sentiment_score, timestamp) VALUES (?, ?, ?, ?, ?)",
        ("u1", "books", "Call me Ace", 0.7, "2024-01-01T00:00:00+00:00"),
    )
    conn.commit()
    conn.close()

    store = InMemoryGraphMemoryStore()
    stats = migrate_sqlite_memories_to_graph(str(db_path), store)

    assert stats["rows_read"] == 1
    assert stats["graph_upserts"] >= 1
    assert retrieve_user_context(store, "u1", limit=5)


def test_canonical_dedup_merges_semantically_identical_facts():
    store = InMemoryGraphMemoryStore()
    triples = [
        {
            "subject_id": "u1",
            "predicate": "likes_hobby",
            "object_value": " Chess ",
            "fact_type": "preference",
            "confidence": 0.8,
        },
        {
            "subject_id": "u1",
            "predicate": "likes_hobby",
            "object_value": "chess",
            "fact_type": "preference",
            "confidence": 0.7,
        },
    ]
    ingest_fact_triples(triples, store, timestamp="2024-01-01T00:00:00+00:00", source_id="dup")

    assert len(store.facts) == 1
    only = next(iter(store.facts.values()))
    assert only.dedup_key


def test_retrieval_interoperability_uses_canonical_fact_shape():
    store = InMemoryGraphMemoryStore()
    ingest_conversation_turns(
        [
            {"user_id": "u9", "text": "My favorite snack is popcorn", "timestamp": "2024-01-01T00:00:00+00:00", "input_id": "turn1"},
        ],
        store,
    )

    context = retrieve_user_context(store, "u9", limit=5)
    assert context
    ev = context[0]
    assert "subject" in ev.attributes
    assert "updated_at" in ev.attributes
