"""Iteration D (M8): Graph-Expansion beim Retrieval.

Diagnose aus LongMemEval-Pilot: multi-session-Fragen (0 %) scheitern, weil die
Evidenz über Sessions verstreut ist — im Wissens-Graph sind die Stücke aber
über gemeinsame Entitäten bereits verbunden. Ein Treffer zieht seine
Nachbar-Segmente mit herein.
"""
import pytest

from semanticswap.memory.retrieval import Retriever
from semanticswap.memory.store import Segment, Store
from semanticswap.memory.vector import NaiveVectorStore
from semanticswap.tokens import TokenCounter


def seg(sid: str, mem: str, text: str) -> Segment:
    return Segment(id=sid, session_id=mem, start_idx=0, end_idx=1, text=text,
                   summary="", priority="high")


def _store_with_graph() -> Store:
    store = Store(":memory:")
    store.add_segment(seg("seg_a", "mem1", "Anna plant eine Reise nach Rom."))
    store.add_segment(seg("seg_b", "mem1", "Anna bucht ein Hotel für die Reise."))
    store.add_segment(seg("seg_c", "mem1", "Bert kauft ein Fahrrad."))
    store.add_segment(seg("seg_x", "mem2", "Anna in einem fremden Speicher."))
    store.add_triples("mem1", "seg_a", [("Anna", "plant", "Reise"),
                                        ("Reise", "ziel", "Rom")])
    store.add_triples("mem1", "seg_b", [("Anna", "bucht", "Hotel"),
                                        ("Hotel", "für", "Reise")])
    store.add_triples("mem1", "seg_c", [("Bert", "kauft", "Fahrrad")])
    store.add_triples("mem2", "seg_x", [("Anna", "ist", "woanders")])
    return store


def test_expand_by_graph_finds_shared_entity_neighbors():
    store = _store_with_graph()
    neighbors = store.expand_by_graph(["seg_a"], "mem1", limit=5)
    ids = [n for n, _ in neighbors]
    assert "seg_b" in ids          # teilt "Anna" und "Reise"
    assert "seg_c" not in ids      # keine gemeinsame Entität
    assert "seg_x" not in ids      # fremder Speicherraum
    assert "seg_a" not in ids      # Seed nicht zurückgeben


def test_expand_ranks_by_shared_entity_count():
    store = _store_with_graph()
    # seg_b teilt zwei Entitäten mit seg_a (Anna, Reise) -> stärkster Nachbar
    neighbors = store.expand_by_graph(["seg_a"], "mem1", limit=5)
    assert neighbors[0][0] == "seg_b"
    assert neighbors[0][1] >= 2


@pytest.mark.asyncio
async def test_retrieval_pulls_graph_neighbors_into_injection(test_config):
    test_config.retrieval.enabled = True
    test_config.retrieval.hybrid = True
    test_config.retrieval.graph_expansion = True
    test_config.embedding.enabled = True
    store = _store_with_graph()
    store.create_session("s1", memory_id="mem1")

    class KwOnlyLLM:  # Embeddings kaputt -> Hybrid nutzt nur FTS, deterministisch
        async def embed(self, model, texts, **kw):
            raise RuntimeError("kein Embedding")

    retr = Retriever(store, NaiveVectorStore(store), KwOnlyLLM(),
                     TokenCounter(use_tiktoken=False), test_config)
    # Query trifft per Volltext seg_a ("Rom"); Graph zieht seg_b (Hotel) mit
    content = await retr.snippets_for_query("s1", "Was ist mit Rom?")
    assert content is not None
    assert "Rom" in content and "Hotel" in content


@pytest.mark.asyncio
async def test_graph_expansion_off_by_default(test_config):
    test_config.retrieval.enabled = True
    test_config.retrieval.hybrid = True
    test_config.embedding.enabled = True
    store = _store_with_graph()
    store.create_session("s1", memory_id="mem1")

    class KwOnlyLLM:
        async def embed(self, model, texts, **kw):
            raise RuntimeError("kein Embedding")

    retr = Retriever(store, NaiveVectorStore(store), KwOnlyLLM(),
                     TokenCounter(use_tiktoken=False), test_config)
    content = await retr.snippets_for_query("s1", "Was ist mit Rom?")
    assert content is not None and "Rom" in content
    assert "Hotel" not in content  # ohne Expansion kein Nachbar
