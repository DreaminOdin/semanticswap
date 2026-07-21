"""Prio 1 (Architektur-Ausbau): Re-Ranker nach dem Hybrid-Retrieval.

Standard-RAG-Lücke: nach der ersten (schnellen, ungenauen) Kandidatensuche
bewertet ein Re-Ranker jedes (Query, Kandidat)-Paar GEMEINSAM und sortiert neu.
Adressiert direkt das gemessene Beifang-Problem (richtiges Snippet ertrinkt).
Hier LLM-Listwise (RankGPT-Stil): kein neues Modell, nutzt das lokale Ollama.
"""
import pytest

from semanticswap.config import AppConfig
from semanticswap.memory.retrieval import Retriever, _parse_rank_order
from semanticswap.memory.store import Segment, Store
from semanticswap.memory.vector import NaiveVectorStore
from semanticswap.tokens import TokenCounter


def test_parse_rank_order_robust():
    assert _parse_rank_order("2, 1, 3", 3) == [1, 0, 2]      # 1-basiert -> 0-basiert
    assert _parse_rank_order("[3] [1]", 3) == [2, 0]         # Klammern egal
    assert _parse_rank_order("Ranking: 2 dann 1", 2) == [1, 0]
    assert _parse_rank_order("Kauderwelsch", 3) is None      # unparsebar -> None
    assert _parse_rank_order("2, 9, 1", 3) == [1, 0]         # out-of-range verworfen


def _cfg() -> AppConfig:
    cfg = AppConfig()
    cfg.retrieval.hybrid = True
    cfg.retrieval.rerank = True
    cfg.retrieval.rerank_backend = "llm"  # diese Datei testet den LLM-Pfad
    cfg.retrieval.rerank_candidates = 10
    cfg.retrieval.top_k = 2
    cfg.embedding.enabled = True
    return cfg


def _store() -> Store:
    store = Store(":memory:")
    store.create_session("s1", memory_id="mem1")
    # gemeinsames Stichwort "Training", damit FTS mehrere Kandidaten liefert
    for i, txt in enumerate([
        "Training: Smalltalk über das Wetter.",
        "Training: Bestzeit im 5-km-Lauf war 25:50.",
        "Training: noch mehr Smalltalk ohne Fakten.",
    ]):
        store.add_segment(Segment(id=f"seg_{i}", session_id="mem1",
                                  start_idx=i, end_idx=i, text=txt,
                                  summary="", priority="high"))
    return store


class RankingLLM:
    """Judge/Reranker-Stub: setzt das faktenhaltige Segment (Index 2, 1-basiert)
    an Position 1, egal wie die Kandidaten reinkamen."""
    def __init__(self):
        self.rerank_calls = 0

    async def embed(self, model, texts, **kw):
        raise RuntimeError("kein Embedding")  # zwingt Hybrid in den FTS-Pfad

    async def complete(self, model, messages, **kw):
        self.rerank_calls += 1
        return {"choices": [{"message": {"content": "2, 1, 3"}}]}


@pytest.mark.asyncio
async def test_rerank_method_reorders_by_llm_ranking():
    # Reorder-Logik direkt: Stub sagt "3, 2, 1" -> Reihenfolge umgekehrt.
    cfg, store = _cfg(), _store()

    class RevLLM(RankingLLM):
        async def complete(self, model, messages, **kw):
            self.rerank_calls += 1
            return {"choices": [{"message": {"content": "3, 2, 1"}}]}

    retr = Retriever(store, NaiveVectorStore(store), RevLLM(),
                     TokenCounter(use_tiktoken=False), cfg)
    hits = [("seg_0", 0.9), ("seg_1", 0.5), ("seg_2", 0.1)]
    reranked = await retr._rerank("egal", hits, "s1")
    assert [h[0] for h in reranked] == ["seg_2", "seg_1", "seg_0"]


@pytest.mark.asyncio
async def test_reranker_invoked_with_multiple_candidates():
    cfg, store = _cfg(), _store()
    llm = RankingLLM()
    retr = Retriever(store, NaiveVectorStore(store), llm,
                     TokenCounter(use_tiktoken=False), cfg)
    content = await retr.snippets_for_query("s1", "Training Bestzeit 5-km-Lauf?")
    assert content is not None
    assert llm.rerank_calls == 1                 # Re-Ranker wurde aufgerufen


@pytest.mark.asyncio
async def test_reranker_fallback_on_unparseable():
    cfg, store = _cfg(), _store()

    class BadLLM(RankingLLM):
        async def complete(self, model, messages, **kw):
            self.rerank_calls += 1
            return {"choices": [{"message": {"content": "ähm keine Ahnung"}}]}

    retr = Retriever(store, NaiveVectorStore(store), BadLLM(),
                     TokenCounter(use_tiktoken=False), cfg)
    hits = [("seg_0", 0.9), ("seg_1", 0.5)]
    reranked = await retr._rerank("egal", hits, "s1")
    assert [h[0] for h in reranked] == ["seg_0", "seg_1"]  # unverändert


@pytest.mark.asyncio
async def test_rerank_off_makes_no_llm_call():
    cfg, store = _cfg(), _store()
    cfg.retrieval.rerank = False
    llm = RankingLLM()
    retr = Retriever(store, NaiveVectorStore(store), llm,
                     TokenCounter(use_tiktoken=False), cfg)
    await retr.snippets_for_query("s1", "5-km-Bestzeit?")
    assert llm.rerank_calls == 0
