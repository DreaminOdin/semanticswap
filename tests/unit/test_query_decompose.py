"""Prio 3 (Architektur-Ausbau): Query-Decomposition gegen multi-session.

Multi-Hop-Fragen ("Wie oft war ich dieses Jahr wandern?") als EINE Suchanfrage
finden wenig. Ein Vor-Schritt zerlegt sie in Teilfragen, sucht jede und
verschmilzt die Kandidaten (RRF). Ersetzt die verworfene Graph-Expansion als
Ansatz für verstreute Evidenz.
"""
import pytest

from semanticswap.config import AppConfig
from semanticswap.memory.retrieval import Retriever, _parse_subqueries
from semanticswap.memory.store import Segment, Store
from semanticswap.memory.vector import NaiveVectorStore
from semanticswap.tokens import TokenCounter


def test_parse_subqueries_robust():
    assert _parse_subqueries("1. Wann A?\n2. Wann B?") == ["Wann A?", "Wann B?"]
    assert _parse_subqueries("- erste\n- zweite") == ["erste", "zweite"]
    assert _parse_subqueries("nur eine Zeile") == ["nur eine Zeile"]
    assert _parse_subqueries("") == []


def _cfg() -> AppConfig:
    cfg = AppConfig()
    cfg.retrieval.hybrid = True
    cfg.retrieval.query_decompose = True
    cfg.retrieval.top_k = 3
    cfg.embedding.enabled = True
    return cfg


def _store() -> Store:
    store = Store(":memory:")
    store.create_session("s1", memory_id="mem1")
    for i, txt in enumerate([
        "Im März war Anna auf dem Brocken wandern.",
        "Im Juli war Anna in den Alpen wandern.",
        "Anna kocht gerne vegetarisch.",
    ]):
        store.add_segment(Segment(id=f"seg_{i}", session_id="mem1",
                                  start_idx=i, end_idx=i, text=txt,
                                  summary="", priority="high"))
    return store


class DecomposingLLM:
    def __init__(self):
        self.decompose_calls = 0

    async def embed(self, model, texts, **kw):
        raise RuntimeError("kein Embedding")  # nur FTS, deterministisch

    async def complete(self, model, messages, **kw):
        self.decompose_calls += 1
        # zerlegt in zwei Teilfragen mit disjunkten Stichwörtern
        return {"choices": [{"message": {"content":
                "1. Brocken wandern\n2. Alpen wandern"}}]}


@pytest.mark.asyncio
async def test_decompose_merges_evidence_from_subqueries():
    cfg, store = _cfg(), _store()
    llm = DecomposingLLM()
    retr = Retriever(store, NaiveVectorStore(store), llm,
                     TokenCounter(use_tiktoken=False), cfg)
    content = await retr.snippets_for_query("s1", "Wo war Anna überall wandern?")
    assert content is not None
    assert llm.decompose_calls == 1
    # beide Wander-Segmente (aus je einer Teilfrage) sind zusammengeführt
    assert "Brocken" in content and "Alpen" in content


@pytest.mark.asyncio
async def test_decompose_off_makes_no_extra_call():
    cfg, store = _cfg(), _store()
    cfg.retrieval.query_decompose = False
    llm = DecomposingLLM()
    retr = Retriever(store, NaiveVectorStore(store), llm,
                     TokenCounter(use_tiktoken=False), cfg)
    await retr.snippets_for_query("s1", "Brocken wandern?")
    assert llm.decompose_calls == 0
