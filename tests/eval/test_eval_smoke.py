"""M5-Smoke: Der Eval-Harness läuft offline durch und die Speicher-Mechanik
liefert die verankerten Fakten über die Kompression hinweg zurück (ADR-007)."""
import pytest

from semanticswap.eval.fakes import OfflineLLM, bow_embed
from semanticswap.eval.run import fake_config
from semanticswap.eval.runner import run_eval
from semanticswap.eval.scenario import build_scenario


def test_bow_embeddings_reflect_word_overlap():
    a = bow_embed("Datenbank Migration Postgres Pooling")
    b = bow_embed("Datenbank Migration Postgres Blocker")
    c = bow_embed("Frontend Redesign Tailwind Komponenten")
    sim = lambda x, y: sum(p * q for p, q in zip(x, y))
    assert sim(a, b) > sim(a, c)


@pytest.mark.asyncio
async def test_offline_eval_end_to_end():
    turns, questions = build_scenario(n_topics=6, filler_sentences=12)
    report = await run_eval(fake_config(), OfflineLLM(), turns, questions)

    # Kompression hat stattgefunden und Speicher wurde aufgebaut
    assert report.archived_upto > 0
    assert report.segments > 0
    assert report.triples > 0
    assert report.archive_ratio is not None and report.archive_ratio > 1.5

    # Upstream-Kontext ist deutlich kleiner als der Client-Verlauf
    assert report.context_ratio is not None and report.context_ratio > 1.2

    # Extraktives Offline-LLM: Treffer nur möglich, wenn das Speichersystem
    # den Fakt geliefert hat (Archiv-Graph, Swap-In oder unkomprimierter Tail)
    assert report.recall_rate >= 0.75

    # Report ist renderbar
    md = report.to_markdown(fake_mode=True)
    assert "Recall-Quote" in md
    assert "Archiv-Kompressions-Ratio" in md
