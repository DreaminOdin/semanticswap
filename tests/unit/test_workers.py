"""M3: Sub-Agenten-Worker und robustes Triple-Parsing."""
import pytest

from semanticswap.compression.workers import (ExtractionWorkers, parse_triples,
                                              strip_reasoning)
from semanticswap.config import SubAgentsConfig


def test_strip_reasoning_removes_think_blocks():
    assert strip_reasoning("<think>lange Überlegung</think>Antwort.") == "Antwort."
    # Reasoning-Modelle ohne öffnendes Tag: alles vor </think> verwerfen
    assert strip_reasoning("Der User fragt nach X...\n</think>\n\nZX-147") == "ZX-147"
    assert strip_reasoning("normale Antwort") == "normale Antwort"


def test_parse_triples_caps_at_limit():
    import json
    raw = json.dumps([{"subject": f"S{i}", "predicate": "p", "object": "O"}
                      for i in range(50)])
    assert len(parse_triples(raw)) == 15


def test_add_triples_is_idempotent_per_segment():
    from semanticswap.memory.store import Store

    store = Store(":memory:")
    store.add_triples("s1", "seg_a", [("A", "p", "B"), ("C", "p", "D")])
    store.add_triples("s1", "seg_a", [("A", "p", "B")])  # Nach-Archivierung
    assert store.get_triples("s1") == [("A", "p", "B")]


def test_parse_triples_plain_json():
    raw = '[{"subject": "A", "predicate": "kennt", "object": "B"}]'
    assert parse_triples(raw) == [("A", "kennt", "B")]


def test_parse_triples_with_surrounding_prose():
    raw = 'Here are the triples:\n[{"subject": "A", "predicate": "p", "object": "B"}]\nDone.'
    assert parse_triples(raw) == [("A", "p", "B")]


def test_parse_triples_invalid_returns_empty():
    assert parse_triples("kein json") == []
    assert parse_triples('{"subject": "kein array"}') == []
    assert parse_triples('[{"subject": "", "predicate": "p", "object": "B"}]') == []


@pytest.mark.asyncio
async def test_process_segment_runs_both_workers(fake_llm, test_config):
    workers = ExtractionWorkers(fake_llm, test_config.sub_agents)
    result = await workers.process_segment("USER: Wir migrieren zu Postgres.")
    assert "Zusammenfassung" in result.summary
    assert ("User", "arbeitet_an", "Projekt X") in result.triples
    models_called = {c["model"] for c in fake_llm.calls}
    assert models_called == {"fake/summary", "fake/entity"}


@pytest.mark.asyncio
async def test_worker_failure_degrades_gracefully(test_config):
    class BrokenLLM:
        async def complete(self, *a, **kw):
            raise RuntimeError("provider down")

    workers = ExtractionWorkers(BrokenLLM(), test_config.sub_agents)
    result = await workers.process_segment("text")
    assert result.summary == ""
    assert result.triples == []
