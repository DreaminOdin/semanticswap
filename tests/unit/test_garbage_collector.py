"""M6.2: Semantischer Garbage Collector - Prioritäten & Deep Archive (ADR-011)."""
import pytest

from semanticswap.compression.workers import ExtractionWorkers, split_priority
from semanticswap.memory.store import Segment, Store
from semanticswap.prompts import build_archive_prompt


def seg(i: int, priority: str) -> Segment:
    return Segment(id=f"seg_{i:03d}", session_id="s1", start_idx=i, end_idx=i,
                   text=f"text {i}", summary=f"summary {i}", priority=priority)


def test_split_priority_variants():
    assert split_priority("PRIORITY: high\nFakten...") == ("high", "Fakten...")
    assert split_priority("priority: LOW. Smalltalk") == ("low", "Smalltalk")
    assert split_priority("ohne marker") == ("high", "ohne marker")
    assert split_priority("") == ("high", "")


@pytest.mark.asyncio
async def test_worker_parses_priority(test_config):
    class LowPrioLLM:
        async def complete(self, model, messages, **kw):
            content = ('PRIORITY: low\nNur Smalltalk.' if "summary" in model else "[]")
            return {"choices": [{"message": {"role": "assistant",
                                             "content": content}}]}

    workers = ExtractionWorkers(LowPrioLLM(), test_config.sub_agents)
    result = await workers.process_segment("USER: wie ist das wetter?")
    assert result.priority == "low"
    assert result.summary == "Nur Smalltalk."


def test_old_low_priority_segments_move_to_deep_archive():
    segments = [seg(1, "low"), seg(2, "high"), seg(3, "low"),
                seg(4, "high"), seg(5, "low")]
    prompt = build_archive_prompt(segments, [], low_priority_visible=1)

    assert "seg_002" in prompt and "seg_004" in prompt   # high bleibt immer
    assert "seg_005" in prompt                            # jüngstes low bleibt sichtbar
    assert "seg_001" not in prompt and "seg_003" not in prompt
    assert "2 low-priority segment(s) moved to deep archive" in prompt


def test_all_high_priority_keeps_everything():
    segments = [seg(i, "high") for i in range(1, 5)]
    prompt = build_archive_prompt(segments, [], low_priority_visible=1)
    assert all(f"seg_{i:03d}" in prompt for i in range(1, 5))
    assert "deep archive" not in prompt


def test_deep_archived_segment_stays_retrievable():
    store = Store(":memory:")
    store.add_segment(seg(1, "low"))
    fetched = store.get_segment("seg_001")
    assert fetched is not None
    assert fetched.priority == "low"
    assert fetched.text == "text 1"
