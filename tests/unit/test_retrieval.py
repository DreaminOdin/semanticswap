"""M4: Swap-In v1 - Vektor-Lookup, Token-Budget, Injection (ADR-008)."""
import pytest

from semanticswap.memory.retrieval import Retriever
from semanticswap.memory.store import Segment, Store
from semanticswap.memory.vector import NaiveVectorStore
from semanticswap.prompts import RETRIEVAL_HEADER
from semanticswap.tokens import TokenCounter

SID = "sess0001"


async def seed(store: Store, vectors: NaiveVectorStore, fake_llm,
               texts: dict[str, str]) -> None:
    store.create_session(session_id=SID)
    for seg_id, text in texts.items():
        store.add_segment(Segment(id=seg_id, session_id=SID, start_idx=1,
                                  end_idx=2, text=text, summary="s"))
        vec = (await fake_llm.embed("fake/embed", [text]))[0]
        vectors.add(seg_id, vec)


def make_retriever(fake_llm, test_config, store=None):
    store = store or Store(":memory:")
    vectors = NaiveVectorStore(store)
    counter = TokenCounter(use_tiktoken=False)
    return Retriever(store, vectors, fake_llm, counter, test_config), store, vectors


def user_msgs(query: str) -> list[dict]:
    return [{"role": "system", "content": "sys"}, {"role": "user", "content": query}]


@pytest.mark.asyncio
async def test_exact_match_ranks_first_and_injects_original(fake_llm, test_config):
    retriever, store, vectors = make_retriever(fake_llm, test_config)
    await seed(store, vectors, fake_llm, {
        "seg_a": "USER: Postgres Migration blockiert durch Connection Pooling.",
        "seg_b": "USER: Frontend Redesign mit Tailwind.",
    })
    query = "USER: Postgres Migration blockiert durch Connection Pooling."
    injection = await retriever.build_injection(SID, user_msgs(query))
    assert injection is not None
    assert injection["role"] == "system"
    assert RETRIEVAL_HEADER in injection["content"]
    assert "Connection Pooling" in injection["content"]
    # exakter Treffer steht vorn
    assert injection["content"].index("seg_a") < injection["content"].index("seg_b")


@pytest.mark.asyncio
async def test_min_score_filters_all(fake_llm, test_config):
    cfg = test_config.model_copy(deep=True)
    cfg.retrieval.min_score = 1.5  # unerreichbar
    retriever, store, vectors = make_retriever(fake_llm, cfg)
    await seed(store, vectors, fake_llm, {"seg_a": "irgendwas"})
    assert await retriever.build_injection(SID, user_msgs("frage")) is None


@pytest.mark.asyncio
async def test_disabled_retrieval_returns_none(fake_llm, test_config):
    cfg = test_config.model_copy(deep=True)
    cfg.retrieval.enabled = False
    retriever, store, vectors = make_retriever(fake_llm, cfg)
    await seed(store, vectors, fake_llm, {"seg_a": "text"})
    assert await retriever.build_injection(SID, user_msgs("text")) is None


@pytest.mark.asyncio
async def test_no_user_message_returns_none(fake_llm, test_config):
    retriever, _, _ = make_retriever(fake_llm, test_config)
    msgs = [{"role": "system", "content": "nur system"}]
    assert await retriever.build_injection(SID, msgs) is None


@pytest.mark.asyncio
async def test_token_budget_truncates(fake_llm, test_config):
    cfg = test_config.model_copy(deep=True)
    cfg.retrieval.max_injection_tokens = 100  # heuristisch ~400 Zeichen
    retriever, store, vectors = make_retriever(fake_llm, cfg)
    long_text = "USER: " + "sehr langer inhalt " * 200  # >> Budget
    await seed(store, vectors, fake_llm, {"seg_a": long_text, "seg_b": long_text + "b"})
    injection = await retriever.build_injection(SID, user_msgs(long_text))
    assert injection is not None
    # nur das erste (gekürzte) Segment passt ins Budget
    assert "seg_b" not in injection["content"]
    assert len(injection["content"]) < len(long_text)


@pytest.mark.asyncio
async def test_embedding_failure_degrades_to_none(test_config):
    class BrokenEmbedder:
        async def embed(self, *a, **kw):
            raise RuntimeError("provider down")

    retriever, store, vectors = make_retriever(BrokenEmbedder(), test_config)
    store.create_session(session_id=SID)
    assert await retriever.build_injection(SID, user_msgs("frage")) is None
