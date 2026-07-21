"""Auswertungs-Metriken in Store.stats() - Basis der Studio-KPIs."""
from semanticswap.memory.store import Segment, Store


def test_stats_includes_archive_metrics():
    store = Store(":memory:")
    store.create_session("s1")
    store.add_segment(Segment(id="seg_1", session_id="s1", start_idx=0, end_idx=3,
                              text="x" * 400, summary="y" * 100, priority="high"))
    store.add_segment(Segment(id="seg_2", session_id="s1", start_idx=4, end_idx=7,
                              text="x" * 600, summary="y" * 150, priority="low"))
    store.set_archive("s1", 8, "z" * 200)

    stats = store.stats()
    assert stats["archived_chars"] == 1000
    assert stats["summary_chars"] == 250
    assert stats["prompt_chars"] == 200
    assert stats["low_priority_segments"] == 1


def test_stats_archive_metrics_empty_db():
    stats = Store(":memory:").stats()
    assert stats["archived_chars"] == 0
    assert stats["summary_chars"] == 0
    assert stats["prompt_chars"] == 0
    assert stats["low_priority_segments"] == 0
