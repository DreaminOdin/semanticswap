"""M3: Swap-Out-Pipeline (Modus B) - Segmentierung, Worker, Persistenz, Archiv."""
import pytest

from semanticswap.compression.pipeline import CompressionPipeline
from semanticswap.compression.workers import ExtractionWorkers
from semanticswap.memory.store import Store
from semanticswap.prompts import ARCHIVE_HEADER
from semanticswap.tokens import TokenCounter


def seed_session(store: Store, n_turns: int = 4) -> str:
    session = store.create_session()
    messages = [{"role": "system", "content": "You are helpful."}]
    for i in range(n_turns):
        messages.append({"role": "user", "content": f"Frage {i}: " + "inhalt " * 30})
        messages.append({"role": "assistant", "content": f"Antwort {i}: " + "detail " * 30})
    store.add_messages(session.id, 0, messages)
    return session.id


@pytest.fixture
def pipeline(fake_llm, test_config):
    store = Store(":memory:")
    workers = ExtractionWorkers(fake_llm, test_config.sub_agents)
    counter = TokenCounter(use_tiktoken=False)
    return CompressionPipeline(store, workers, counter, test_config, llm=fake_llm), store


@pytest.mark.asyncio
async def test_compress_creates_segments_and_archive(pipeline):
    pipe, store = pipeline
    sid = seed_session(store)

    created = await pipe.compress_session(sid)
    assert created > 0

    session = store.get_session(sid)
    # keep_recent_messages=2 -> die letzten 2 Messages bleiben unkomprimiert
    assert session.archived_upto == store.message_count(sid) - 2
    assert ARCHIVE_HEADER in session.archive_prompt
    assert "#seg_" in session.archive_prompt

    segments = store.get_segments(sid)
    assert segments
    # System-Prompt (idx 0) wird nie archiviert
    assert all(seg.start_idx >= 1 for seg in segments)
    assert store.get_triples(sid)
    assert store.all_embeddings(sid)  # Embeddings wurden geschrieben


@pytest.mark.asyncio
async def test_compress_is_incremental_and_idempotent(pipeline):
    pipe, store = pipeline
    sid = seed_session(store)
    await pipe.compress_session(sid)
    first_count = len(store.get_segments(sid))

    # Ohne neue Messages: nichts zu tun
    assert await pipe.compress_session(sid) == 0
    assert len(store.get_segments(sid)) == first_count

    # Neue Messages -> nur der neue Teil wird archiviert
    n = store.message_count(sid)
    store.add_messages(sid, n, [
        {"role": "user", "content": "Neue Frage " + "mehr " * 40},
        {"role": "assistant", "content": "Neue Antwort " + "mehr " * 40},
        {"role": "user", "content": "Noch eine Frage"},
        {"role": "assistant", "content": "Noch eine Antwort"},
    ])
    created = await pipe.compress_session(sid)
    assert created > 0
    new_segments = store.get_segments(sid)[first_count:]
    assert all(seg.start_idx >= n - 2 for seg in new_segments)


@pytest.mark.asyncio
async def test_short_history_is_not_compressed(pipeline):
    pipe, store = pipeline
    session = store.create_session()
    store.add_messages(session.id, 0, [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hallo"},
        {"role": "assistant", "content": "hi"},
    ])
    assert await pipe.compress_session(session.id) == 0
    assert store.get_session(session.id).archived_upto == 0
