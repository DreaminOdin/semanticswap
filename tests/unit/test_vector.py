"""M3/M4-Vorbereitung: naiver Vektor-Store (ADR-004)."""
from semanticswap.memory.store import Segment, Store
from semanticswap.memory.vector import NaiveVectorStore


def _seed_segment(store: Store, seg_id: str, session_id: str = "s1") -> None:
    store.add_segment(Segment(id=seg_id, session_id=session_id, start_idx=0,
                              end_idx=1, text="t", summary="s"))


def test_search_ranks_by_cosine_similarity():
    store = Store(":memory:")
    vectors = NaiveVectorStore(store)
    _seed_segment(store, "seg_a")
    _seed_segment(store, "seg_b")
    vectors.add("seg_a", [1.0, 0.0, 0.0])
    vectors.add("seg_b", [0.0, 1.0, 0.0])

    results = vectors.search([0.9, 0.1, 0.0], top_k=2)
    assert results[0][0] == "seg_a"
    assert results[0][1] > results[1][1]


def test_search_filters_by_session():
    store = Store(":memory:")
    vectors = NaiveVectorStore(store)
    _seed_segment(store, "seg_a", session_id="s1")
    _seed_segment(store, "seg_b", session_id="s2")
    vectors.add("seg_a", [1.0, 0.0])
    vectors.add("seg_b", [1.0, 0.0])

    results = vectors.search([1.0, 0.0], session_id="s1")
    assert [r[0] for r in results] == ["seg_a"]


def test_empty_store_returns_empty():
    vectors = NaiveVectorStore(Store(":memory:"))
    assert vectors.search([1.0, 0.0]) == []


def test_batched_search_matches_reference_on_many_vectors():
    # Prio 5: numpy-vektorisierte Suche muss dasselbe Ranking liefern wie die
    # naive Referenz (nur schneller).
    import math
    import random

    store = Store(":memory:")
    vectors = NaiveVectorStore(store)
    dim = 16
    rng = random.Random(42)
    data = {}
    for i in range(200):
        _seed_segment(store, f"seg_{i}")
        v = [rng.gauss(0, 1) for _ in range(dim)]
        data[f"seg_{i}"] = v
        vectors.add(f"seg_{i}", v)
    q = [rng.gauss(0, 1) for _ in range(dim)]

    def cos(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na and nb else 0.0

    ref = sorted(data.items(), key=lambda kv: cos(q, kv[1]), reverse=True)[:5]
    got = vectors.search(q, top_k=5)
    assert [g[0] for g in got] == [r[0] for r in ref]
