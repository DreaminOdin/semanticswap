"""ADR-013: LongMemEval-Runner — Mechanik offline (OfflineLLM, Substring-Judge)."""
import json

import pytest

from semanticswap.eval import longmemeval as lme
from semanticswap.eval.fakes import OfflineLLM
from semanticswap.eval.run import fake_config

FILLER = [
    {"role": "user", "content": "Wie war das Wetter gestern eigentlich?"},
    {"role": "assistant", "content": "Sonnig mit ein paar Wolken."},
    {"role": "user", "content": "Danke dir, nur Smalltalk heute."},
    {"role": "assistant", "content": "Gerne, bis später!"},
]


def _fixture(tmp_path):
    data = [
        {
            "question_id": "q1",
            "question_type": "single-session-user",
            "question": "Wie lautet der Referenzcode für Projekt Adler?",
            "answer": "ADL-042",
            "question_date": "2026/07/18 (Sat)",
            "haystack_dates": ["2026/07/01 (Wed)", "2026/07/02 (Thu)"],
            "haystack_sessions": [
                [
                    {"role": "user", "content": "Notiere bitte: Der "
                     "Referenzcode für Projekt Adler lautet ADL-042."},
                    {"role": "assistant", "content": "Notiert."},
                ] + FILLER,
                FILLER,
            ],
        },
        {
            "question_id": "q2_abs",
            "question_type": "abstention",
            "question": "Wie lautet der Referenzcode für Projekt Falke?",
            "answer": "",
            "question_date": "2026/07/18 (Sat)",
            "haystack_dates": ["2026/07/03 (Fri)"],
            "haystack_sessions": [FILLER],
        },
        {
            "question_id": "q3",
            "question_type": "multi-session",
            "question": "Wie lautet der Referenzcode für Projekt Biber?",
            "answer": "BIB-007",
            "question_date": "2026/07/18 (Sat)",
            "haystack_dates": ["2026/07/04 (Sat)", "2026/07/05 (Sun)"],
            "haystack_sessions": [
                FILLER,
                [
                    {"role": "user", "content": "Der Referenzcode für "
                     "Projekt Biber lautet BIB-007."},
                    {"role": "assistant", "content": "Habe ich mir gemerkt."},
                ] + FILLER,
            ],
        },
    ]
    path = tmp_path / "lme.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def test_loader_filters_and_stratifies(tmp_path):
    path = _fixture(tmp_path)
    all_inst, total = lme.load_instances(path)
    assert total == 3 and len(all_inst) == 3

    limited, total = lme.load_instances(path, limit=2)
    assert total == 3 and len(limited) == 2
    # stratifiziert: zwei verschiedene Typen, nicht zweimal derselbe
    assert len({i.question_type for i in limited}) == 2

    typed, _ = lme.load_instances(path, types=["abstention"])
    assert [i.question_id for i in typed] == ["q2_abs"]


def test_flatten_history_embeds_session_dates(tmp_path):
    instances, _ = lme.load_instances(_fixture(tmp_path))
    history = lme.flatten_history(instances[0])
    assert history[0]["content"].startswith("(session date: 2026/07/01")
    # Datum nur im jeweils ersten User-Turn der Session
    assert sum("session date" in m["content"] for m in history) == 2


def test_abstention_detection():
    inst, _ = (lme.LMEInstance("x_abs", "abstention", "?", "", "", [], []), None)
    assert inst.is_abstention
    assert not lme.LMEInstance("x", "t", "?", "", "", [], []).is_abstention


@pytest.mark.asyncio
async def test_llm_judge_passes_api_base(tmp_path):
    # Regression 2026-07-18: ohne api_base landet der Judge beim
    # Default-Ollama (localhost) statt beim konfigurierten Backend.
    captured = {}

    class StubLLM:
        async def complete(self, model, messages, **kwargs):
            captured.update(kwargs, model=model)
            return {"choices": [{"message": {"content": "yes"}}]}

    inst = lme.LMEInstance("q", "t", "Frage?", "Antwort", "", [], [])
    judge = lme.make_llm_judge(StubLLM(), "ollama/qwen2.5:7b",
                               api_base="http://127.0.0.1:11436")
    assert await judge(inst, "Antwort") is True
    assert captured["api_base"] == "http://127.0.0.1:11436"
    assert captured["model"] == "ollama/qwen2.5:7b"


@pytest.mark.asyncio
async def test_ingestion_cache_reused_across_runs(tmp_path):
    # Coarse-to-fine-Iterationen (PO 2026-07-19): Die Ingestion (~20 min/
    # Instanz) dominiert die Laufzeit, ist aber für Retrieval-Experimente
    # identisch. Mit --workdir wird sie gecacht; der zweite Lauf macht
    # keine Worker-Calls mehr und lässt den Cache unangetastet.
    instances, total = lme.load_instances(_fixture(tmp_path))
    cache = tmp_path / "cache"

    async def judge(inst, hypothesis):
        return lme.contains_judge(inst, hypothesis)

    llm1 = OfflineLLM()
    r1 = await lme.run_benchmark(fake_config(), llm1, instances, total, judge,
                                 judge_description="t", dataset_name="f",
                                 workdir=str(cache))
    worker_calls_run1 = sum(1 for c in llm1.calls
                            if "summary" in c["model"] or "entity" in c["model"])
    assert worker_calls_run1 > 0

    import hashlib as _h

    cache_hash_before = _h.md5((cache / "q1.db").read_bytes()).hexdigest()

    llm2 = OfflineLLM()
    r2 = await lme.run_benchmark(fake_config(), llm2, instances, total, judge,
                                 judge_description="t", dataset_name="f",
                                 workdir=str(cache))
    worker_calls_run2 = sum(1 for c in llm2.calls
                            if "summary" in c["model"] or "entity" in c["model"])
    # Lauf 2 macht nur noch die Frage-Phasen-Kompression, keine Ingestion
    # (die Fixture-Ingestion erzeugt mehr Worker-Calls als die Frage-Phase)
    assert worker_calls_run2 < worker_calls_run1
    assert [x.hit for x in r2.results] == [x.hit for x in r1.results]
    # Cache-DB ist byte-identisch geblieben (Frage-Phase lief auf einer Kopie)
    assert _h.md5((cache / "q1.db").read_bytes()).hexdigest() == cache_hash_before
    assert (cache / "q1.ingest.json").exists()


@pytest.mark.asyncio
async def test_ingestion_cache_invalidated_on_compression_change(tmp_path):
    # Kompressions-relevante Config-Änderungen müssen neu ingestieren.
    instances, total = lme.load_instances(_fixture(tmp_path))
    cache = tmp_path / "cache"

    async def judge(inst, hypothesis):
        return lme.contains_judge(inst, hypothesis)

    await lme.run_benchmark(fake_config(), OfflineLLM(), instances, total,
                            judge, judge_description="t", dataset_name="f",
                            workdir=str(cache))
    cfg2 = fake_config()
    cfg2.storage.chunk_size = 120  # kompressions-relevant
    llm2 = OfflineLLM()
    await lme.run_benchmark(cfg2, llm2, instances, total, judge,
                            judge_description="t", dataset_name="f",
                            workdir=str(cache))
    worker_calls = sum(1 for c in llm2.calls
                       if "summary" in c["model"] or "entity" in c["model"])
    assert worker_calls > 0  # Cache wurde korrekt invalidiert


@pytest.mark.asyncio
async def test_benchmark_survives_instance_error_and_saves(tmp_path, monkeypatch):
    # Regression 2026-07-18: Ein Ollama-Timeout in Instanz 4 riss den ganzen
    # Pilot ab und verwarf 3 fertige Ergebnisse. Fehler müssen übersprungen
    # und der Zwischenstand nach jeder Instanz gespeichert werden.
    instances, total = lme.load_instances(_fixture(tmp_path))

    real_run = lme.run_instance

    async def flaky(cfg, llm, inst, judge, workdir):
        if inst.question_id == "q2_abs":
            raise RuntimeError("simulierter Timeout")
        return await real_run(cfg, llm, inst, judge, workdir)

    monkeypatch.setattr(lme, "run_instance", flaky)

    async def judge(inst, hypothesis):
        return lme.contains_judge(inst, hypothesis)

    save = tmp_path / "out.json"
    report = await lme.run_benchmark(
        fake_config(), OfflineLLM(), instances, total, judge,
        judge_description="Substring (Test)", dataset_name="fixture",
        save_path=str(save))

    assert len(report.results) == 2 and len(report.errors) == 1
    assert "q2_abs" in report.errors[0]
    assert "übersprungen" in report.to_markdown()

    # Zwischenstand ist geladen und resumefähig
    prior = lme.load_prior_report(str(save))
    assert {r.question_id for r in prior.results} == {"q1", "q3"}


@pytest.mark.asyncio
async def test_benchmark_offline_end_to_end(tmp_path):
    instances, total = lme.load_instances(_fixture(tmp_path))

    async def judge(inst, hypothesis):
        return lme.contains_judge(inst, hypothesis)

    report = await lme.run_benchmark(
        fake_config(), OfflineLLM(), instances, total, judge,
        judge_description="Substring (Test)", dataset_name="fixture")

    assert len(report.results) == 3
    # Fakten wurden archiviert und über das Gedächtnis beantwortet
    fact_results = [r for r in report.results if r.question_id in ("q1", "q3")]
    assert all(r.segments > 0 for r in fact_results)
    assert all(r.hit for r in fact_results)
    # Token-Metriken werden erhoben (Ersparnis-Aussagen erst ab realer
    # Historien-Größe sinnvoll — bei Mini-Fixtures ist der ARCHIVE-Prompt
    # größer als das Original)
    assert all(r.upstream_tokens > 0 and r.full_tokens > 0
               for r in fact_results)
    # Abstention: Offline-LLM verneint mangels Fakt -> korrekt abgelehnt
    abs_result = next(r for r in report.results if r.question_id == "q2_abs")
    assert abs_result.hit

    md = report.to_markdown()
    assert "QA-Accuracy" in md and "abstention" in md
    per_type = report.per_type()
    assert per_type["single-session-user"] == (1, 1)
