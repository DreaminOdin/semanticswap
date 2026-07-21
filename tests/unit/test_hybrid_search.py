"""Iteration A (M8): Hybrid-Retrieval — FTS5-Volltext + Vektoren via RRF.

Diagnose aus LongMemEval-Pilot #1: rein semantische Suche verfehlt exakte
Zeichenketten (Namen, Zahlen, Codes). FTS5 ist in SQLite eingebaut — keine
neue Abhängigkeit.
"""
import pytest

from semanticswap.memory.retrieval import Retriever, _rrf_merge
from semanticswap.memory.store import Segment, Store
from semanticswap.memory.vector import NaiveVectorStore
from semanticswap.tokens import TokenCounter


def seg(i: int, session: str, text: str) -> Segment:
    return Segment(id=f"seg_{i}", session_id=session, start_idx=i, end_idx=i,
                   text=text, summary="", priority="high")


def test_keyword_search_finds_exact_codes_and_respects_session():
    store = Store(":memory:")
    store.add_segment(seg(1, "mem1", "Der Referenzcode lautet XQZ-999."))
    store.add_segment(seg(2, "mem1", "Heute war das Wetter schön."))
    store.add_segment(seg(3, "mem2", "XQZ-999 gehört zu einem anderen Speicher."))

    hits = store.keyword_search("Wie lautet der Code XQZ-999?", 5, "mem1")
    assert hits and hits[0][0] == "seg_1"
    assert all(h[0] != "seg_3" for h in hits)  # fremder Speicherraum bleibt außen vor


def test_fts_backfill_for_existing_dbs(tmp_path):
    # Alt-Datenbanken (z. B. der Ingestion-Cache) haben noch keinen
    # FTS-Index — beim Öffnen wird er aufgefüllt.
    db = tmp_path / "old.db"
    s1 = Store(db)
    s1.add_segment(seg(1, "m", "Backfill-Test mit QQTOKEN drin."))
    s1._conn.execute("DELETE FROM segments_fts")  # simuliert Alt-DB
    s1._conn.commit()
    s1.close()

    s2 = Store(db)
    hits = s2.keyword_search("QQTOKEN", 5, "m")
    s2.close()
    assert hits and hits[0][0] == "seg_1"


def test_rrf_merge_prefers_items_in_both_lists():
    fused = _rrf_merge([["a", "b", "c"], ["c", "d"]], top_k=3)
    assert fused[0] == "c"
    assert set(fused) <= {"a", "b", "c", "d"}


@pytest.mark.asyncio
async def test_retriever_emits_search_event_for_flowchart(test_config):
    # Live-Observability (ADR-012-Nachtrag): Die Speicher-Suche meldet sich
    # auf dem Event-Bus, damit der Retrieval-Knoten im Flowchart pulsiert.
    from semanticswap.events import EventBus

    test_config.retrieval.hybrid = True
    test_config.retrieval.enabled = True
    test_config.embedding.enabled = True
    bus = EventBus()
    store = Store(":memory:")
    session = store.create_session("s1")
    store.add_segment(seg(1, session.memory_id, "Der Wert ist QQCODE-7."))

    class BrokenEmbedLLM:
        async def embed(self, model, texts, **kw):
            raise RuntimeError("down")

    retr = Retriever(store, NaiveVectorStore(store), BrokenEmbedLLM(),
                     TokenCounter(use_tiktoken=False), test_config, bus=bus)
    await retr.snippets_for_query("s1", "Was ist QQCODE-7?")
    search_events = [e for e in bus.history if e["type"] == "retrieval_search"]
    assert search_events
    assert search_events[0]["mode"] == "hybrid"
    assert search_events[0]["kw"] >= 1


@pytest.mark.asyncio
async def test_hybrid_degrades_to_keyword_when_embedding_fails(test_config):
    # Rein vektorbasiert wäre das ein Totalausfall (None); Hybrid liefert
    # weiterhin Stichwort-Treffer.
    test_config.retrieval.hybrid = True
    test_config.retrieval.enabled = True
    test_config.embedding.enabled = True
    store = Store(":memory:")
    session = store.create_session("s1")
    store.add_segment(seg(1, session.memory_id, "Der Wert ist QQCODE-7."))

    class BrokenEmbedLLM:
        async def embed(self, model, texts, **kw):
            raise RuntimeError("Embedding-Backend down")

    retr = Retriever(store, NaiveVectorStore(store), BrokenEmbedLLM(),
                     TokenCounter(use_tiktoken=False), test_config)
    content = await retr.snippets_for_query("s1", "Was ist QQCODE-7?")
    assert content is not None and "QQCODE-7" in content
