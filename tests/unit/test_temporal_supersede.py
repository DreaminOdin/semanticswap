"""Iteration C (M8): temporale Tripel — neuerer Fakt verdrängt älteren.

Diagnose LongMemEval: knowledge-update-Fragen scheitern, weil ein alter Fakt
neben dem neuen im Archiv steht ("wohnt in Berlin" UND "wohnt in München") und
das Modell rät. Fix: bei gleichem Subjekt+Prädikat gewinnt das Tripel aus dem
späteren Segment (höhere start_idx = später im Gespräch).
"""
from semanticswap.memory.store import Segment, Store
from semanticswap.prompts import build_archive_prompt


def seg(mem: str, idx: int) -> Segment:
    return Segment(id=f"seg_{idx}", session_id=mem, start_idx=idx, end_idx=idx,
                   text=f"text {idx}", summary=f"summary {idx}", priority="high")


def test_get_triples_with_recency_orders_by_segment():
    store = Store(":memory:")
    store.add_segment(seg("m", 2))
    store.add_segment(seg("m", 8))
    store.add_triples("m", "seg_2", [("Anna", "wohnt_in", "Berlin")])
    store.add_triples("m", "seg_8", [("Anna", "wohnt_in", "München")])
    rows = store.get_triples_with_recency("m")
    # (subject, predicate, object, recency=start_idx)
    assert ("Anna", "wohnt_in", "Berlin", 2) in rows
    assert ("Anna", "wohnt_in", "München", 8) in rows


def test_supersede_keeps_newest_per_subject_predicate():
    triples = [("Anna", "wohnt_in", "Berlin", 2),
               ("Anna", "wohnt_in", "München", 8),
               ("Anna", "mag", "Tee", 3)]
    prompt = build_archive_prompt([], triples, temporal_supersede=True)
    assert "München" in prompt          # neuerer Wohnort bleibt
    assert "Berlin" not in prompt        # alter verdrängt
    assert "Tee" in prompt               # anderes Prädikat unberührt


def test_supersede_off_keeps_both():
    triples = [("Anna", "wohnt_in", "Berlin", 2),
               ("Anna", "wohnt_in", "München", 8)]
    prompt = build_archive_prompt([], triples, temporal_supersede=False)
    assert "Berlin" in prompt and "München" in prompt


def test_supersede_tolerates_plain_triples_without_recency():
    # Rückwärtskompatibel: 3er-Tupel (ohne Recency) funktionieren weiter
    prompt = build_archive_prompt([], [("A", "p", "B")], temporal_supersede=True)
    assert "(A) -> p -> (B)" in prompt
