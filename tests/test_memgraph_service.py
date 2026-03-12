import os

import pytest

from deepthought.memory import MemoryLifecyclePolicy, MemoryTier
from deepthought.memory.graph import InMemoryGraphMemoryStore, retrieve_user_context
from deepthought.memory.graph.store import GraphEvidence, Provenance
from tests.helpers import memgraph_available


def test_memory_lifecycle_policy_scores_salient_events_higher():
    policy = MemoryLifecyclePolicy()

    low = policy.score_event("weather is okay")
    high = policy.score_event(
        "My favorite editor is vim and I will finish this TODO later"
    )

    assert high.salience > low.salience
    assert high.tier == MemoryTier.LONG_TERM
    assert low.tier == MemoryTier.EPHEMERAL
    assert "user_preference" in high.reason_tags
    assert "unresolved_task" in high.reason_tags


def test_retrieval_prioritizes_summary_evidence_over_raw_logs():
    store = InMemoryGraphMemoryStore()
    summary = GraphEvidence(
        evidence_id="s1",
        summary="summary: prefers tea over coffee",
        entity_id="u1",
        relation_type="summary",
        score=0.7,
        confidence=0.8,
        provenance=Provenance(source="consolidation"),
        attributes={
            "is_summary": True,
            "salience": 0.9,
            "memory_tier": "long_term",
            "user_scoped": True,
        },
    )
    raw = GraphEvidence(
        evidence_id="r1",
        summary="chat turn: tea is nice",
        entity_id="u1",
        relation_type="observation",
        score=0.85,
        confidence=0.9,
        provenance=Provenance(source="raw"),
        attributes={"salience": 0.1, "memory_tier": "ephemeral", "user_scoped": True},
    )

    store.retrieve_user_evidence = lambda user_id, limit=10: [raw, summary]  # type: ignore[method-assign]
    ordered = retrieve_user_context(store, "u1", limit=2)

    assert ordered[0].summary.startswith("summary:")


@pytest.mark.memgraph
def test_memgraph_running():
    pymemgraph = pytest.importorskip("pymemgraph")
    if not memgraph_available():
        pytest.skip("Memgraph not available")
    mg = pymemgraph.Memgraph(
        host=os.getenv("MG_HOST", "localhost"),
        port=int(os.getenv("MG_PORT", 7687)),
        username=os.getenv("MG_USER", "memgraph"),
        password=os.getenv("MG_PASSWORD", "memgraph"),
    )
    result = mg.execute("RETURN 1 AS num;")
    mg.close()
    row = result[0] if result else None
    assert row and (row[0] == 1 if isinstance(row, tuple) else row.get("num") == 1)
