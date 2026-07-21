"""Cross-Encoder-Re-Ranker (ONNX, echtes Werkzeug statt LLM-Listwise).

Der LLM-Listwise-Re-Ranker scheiterte am kleinen Modell (nur "10,1"). Ein
Cross-Encoder bepunktet (Query, Passage)-Paare DIREKT — verlässlich. ONNX via
fastembed (Apache-2.0, kein PyTorch).
"""
import pytest

from semanticswap.config import AppConfig
from semanticswap.memory.retrieval import Retriever
from semanticswap.memory.store import Segment, Store
from semanticswap.memory.vector import NaiveVectorStore
from semanticswap.tokens import TokenCounter


class StubCrossEncoder:
    """Injizierbarer Scorer: höhere Zahl im Text = relevanter."""
    def __init__(self):
        self.calls = 0

    def score(self, query: str, docs: list[str]) -> list[float]:
        self.calls += 1
        # Score = Länge des Texts als Platzhalter-Relevanz (deterministisch)
        return [float(len(d)) for d in docs]


def _cfg() -> AppConfig:
    cfg = AppConfig()
    cfg.retrieval.hybrid = True
    cfg.retrieval.rerank = True
    cfg.retrieval.rerank_backend = "cross_encoder"
    cfg.retrieval.rerank_candidates = 10
    cfg.retrieval.top_k = 2
    cfg.embedding.enabled = True
    return cfg


def _store() -> Store:
    store = Store(":memory:")
    store.create_session("s1", memory_id="mem1")
    for i, txt in enumerate(["Training kurz.", "Training mittellang hier.",
                             "Training das mit Abstand längste Segment von allen."]):
        store.add_segment(Segment(id=f"seg_{i}", session_id="mem1",
                                  start_idx=i, end_idx=i, text=txt,
                                  summary="", priority="high"))
    return store


@pytest.mark.asyncio
async def test_cross_encoder_reorders_by_score():
    cfg, store = _cfg(), _store()

    class EmbedFailLLM:
        async def embed(self, model, texts, **kw):
            raise RuntimeError("kein Embedding")

    ce = StubCrossEncoder()
    retr = Retriever(store, NaiveVectorStore(store), EmbedFailLLM(),
                     TokenCounter(use_tiktoken=False), cfg, reranker=ce)
    hits = [("seg_0", 0.9), ("seg_1", 0.5), ("seg_2", 0.1)]
    reranked = await retr._rerank("Training?", hits, "s1")
    # längster Text (seg_2) zuerst, dann seg_1, dann seg_0
    assert [h[0] for h in reranked] == ["seg_2", "seg_1", "seg_0"]
    assert ce.calls == 1


@pytest.mark.asyncio
async def test_cross_encoder_no_llm_call_for_ranking():
    # Anders als LLM-Listwise ruft der Cross-Encoder KEIN Chat-Modell auf.
    cfg, store = _cfg(), _store()

    class CountingLLM:
        def __init__(self): self.completes = 0
        async def embed(self, model, texts, **kw): raise RuntimeError("x")
        async def complete(self, model, messages, **kw):
            self.completes += 1
            return {"choices": [{"message": {"content": "1"}}]}

    llm = CountingLLM()
    retr = Retriever(store, NaiveVectorStore(store), llm,
                     TokenCounter(use_tiktoken=False), cfg,
                     reranker=StubCrossEncoder())
    await retr.snippets_for_query("s1", "Training?")
    assert llm.completes == 0  # Cross-Encoder ersetzt den LLM-Ranking-Call


@pytest.mark.asyncio
async def test_llm_backend_still_available():
    # Backend "llm" nutzt weiter den Chat-Weg (Rückwärtskompatibilität).
    cfg, store = _cfg(), _store()
    cfg.retrieval.rerank_backend = "llm"

    class RankLLM:
        def __init__(self): self.completes = 0
        async def embed(self, model, texts, **kw): raise RuntimeError("x")
        async def complete(self, model, messages, **kw):
            self.completes += 1
            return {"choices": [{"message": {"content": "2, 1, 3"}}]}

    llm = RankLLM()
    retr = Retriever(store, NaiveVectorStore(store), llm,
                     TokenCounter(use_tiktoken=False), cfg)
    hits = [("seg_0", 0.9), ("seg_1", 0.5), ("seg_2", 0.1)]
    await retr._rerank("q", hits, "s1")
    assert llm.completes == 1  # LLM-Backend ruft das Chat-Modell
