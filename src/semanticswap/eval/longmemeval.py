"""LongMemEval-Runner (ADR-013): externer Benchmark, komplett lokal.

Ablauf pro Instanz:
1. Haystack-Sessions werden direkt in den Store geschrieben und über die
   CompressionPipeline archiviert (Ingestion ohne LLM-Antwort pro Turn).
2. Die Frage läuft als regulärer stateless Request durch das Gateway
   (volle Historie + Frage; virtuelles Pruning ersetzt den archivierten
   Teil durch den ARCHIVE-Prompt, Swap-In/Active Retrieval greifen).
3. Ein lokaler LLM-Judge (oder Substring-Judge im Offline-Modus) bewertet
   die Antwort gegen die Referenz.

Datensatz (MIT-Lizenz): https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
Ablage lokal unter data/benchmarks/ (nicht eingecheckt, ADR-013).
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

import httpx

from ..compression.pipeline import CompressionPipeline
from ..compression.workers import ExtractionWorkers
from ..config import AppConfig, load_config
from ..gateway import create_app
from ..llm import LLMClient
from ..memory.store import Store
from ..tokens import TokenCounter, plain_text
from .fakes import OfflineLLM
from .run import fake_config

log = logging.getLogger("semanticswap.eval.longmemeval")

ABSTENTION_SUFFIX = "_abs"

JUDGE_PROMPT = """You are grading an answer from a memory-augmented assistant.

Question: {question}
Reference answer: {answer}
Model answer: {hypothesis}

Does the model answer contain the same information as the reference answer?
Minor wording differences are fine; the facts must match.
Reply with exactly one word: yes or no."""

JUDGE_PROMPT_ABSTENTION = """You are grading an answer from a memory-augmented assistant.
The question is UNANSWERABLE from the conversation history; the correct
behavior is to say so (or admit not knowing) instead of guessing.

Question: {question}
Model answer: {hypothesis}

Did the model correctly decline to answer or state that it does not know?
Reply with exactly one word: yes or no."""


@dataclass
class LMEInstance:
    question_id: str
    question_type: str
    question: str
    answer: str
    question_date: str
    sessions: list[list[dict]]
    dates: list[str]

    @property
    def is_abstention(self) -> bool:
        return self.question_id.endswith(ABSTENTION_SUFFIX)


@dataclass
class LMEResult:
    question_id: str
    question_type: str
    question: str
    expected: str
    hypothesis: str
    hit: bool
    segments: int
    full_tokens: int
    upstream_tokens: int
    ingest_seconds: float
    answer_seconds: float


@dataclass
class LMEReport:
    dataset: str
    total_in_dataset: int
    results: list[LMEResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    judge_description: str = ""

    @property
    def accuracy(self) -> float:
        return (sum(r.hit for r in self.results) / len(self.results)
                if self.results else 0.0)

    def per_type(self) -> dict[str, tuple[int, int]]:
        """{Typ: (Treffer, Anzahl)} — Stichprobengröße immer mit ausweisen."""
        out: dict[str, tuple[int, int]] = {}
        for r in self.results:
            hits, n = out.get(r.question_type, (0, 0))
            out[r.question_type] = (hits + int(r.hit), n + 1)
        return out

    @property
    def avg_savings(self) -> float | None:
        pairs = [(r.full_tokens, r.upstream_tokens)
                 for r in self.results if r.upstream_tokens > 0]
        if not pairs:
            return None
        return sum(f for f, _ in pairs) / max(1, sum(u for _, u in pairs))

    def to_markdown(self) -> str:
        lines = [
            "# LongMemEval-Report (SemanticSwap)",
            "",
            f"- Instanzen: **{len(self.results)}** von {self.total_in_dataset} "
            f"im Datensatz (Teil-Läufe sind stratifiziert, ADR-013)",
            f"- Judge: {self.judge_description} — NICHT identisch mit dem "
            f"GPT-4o-Judge des Papers; Zahlen als 'self-reported' labeln",
            f"- **QA-Accuracy: {self.accuracy:.1%}**",
        ]
        if self.errors:
            lines.append(f"- ⚠️ {len(self.errors)} Instanz(en) mit Fehler "
                         f"übersprungen (zählen nicht in die Accuracy): "
                         + "; ".join(self.errors[:5]))
        if self.avg_savings is not None:
            lines.append(f"- Kontext-Ersparnis (Voll-Historie vs. Upstream): "
                         f"**{self.avg_savings:.2f}x**")
        lines += ["", "| Fähigkeits-Typ | Treffer | n | Accuracy |",
                  "|---|---|---|---|"]
        for qtype, (hits, n) in sorted(self.per_type().items()):
            lines.append(f"| {qtype} | {hits} | {n} | {hits / n:.1%} |")
        return "\n".join(lines)


def load_instances(path: str | Path, limit: int | None = None,
                   types: list[str] | None = None) -> tuple[list[LMEInstance], int]:
    """Lädt Instanzen; Teil-Auswahl stratifiziert über die Frage-Typen und
    wird sichtbar geloggt (keine stillen Caps, ADR-013)."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    instances = [
        LMEInstance(
            question_id=str(item["question_id"]),
            question_type=str(item.get("question_type", "unknown")),
            question=str(item["question"]),
            answer=str(item.get("answer", "")),
            question_date=str(item.get("question_date", "")),
            sessions=item.get("haystack_sessions", []),
            dates=[str(d) for d in item.get("haystack_dates", [])],
        )
        for item in raw
    ]
    total = len(instances)
    if types:
        instances = [i for i in instances if i.question_type in types]
        log.warning("Typ-Filter %s: %d von %d Instanzen bleiben",
                    types, len(instances), total)
    if limit is not None and limit < len(instances):
        by_type: dict[str, list[LMEInstance]] = {}
        for inst in instances:
            by_type.setdefault(inst.question_type, []).append(inst)
        picked: list[LMEInstance] = []
        while len(picked) < limit and any(by_type.values()):
            for qtype in sorted(by_type):
                if by_type[qtype] and len(picked) < limit:
                    picked.append(by_type[qtype].pop(0))
        log.warning("Stichprobe: %d von %d Instanzen (stratifiziert über %d Typen)",
                    len(picked), len(instances), len(by_type))
        instances = picked
    return instances, total


def flatten_history(inst: LMEInstance) -> list[dict]:
    """Haystack-Sessions -> eine stateless Client-Historie. Das Sitzungsdatum
    wird in den ersten User-Turn jeder Session eingebettet (Zeitbezug für
    temporal reasoning, wie im Paper üblich)."""
    msgs: list[dict] = []
    dates = inst.dates + [""] * (len(inst.sessions) - len(inst.dates))
    for date, session in zip(dates, inst.sessions):
        dated = False
        for turn in session:
            role = str(turn.get("role", "user"))
            content = plain_text(turn.get("content"))
            if role == "user" and not dated and date:
                content = f"(session date: {date}) {content}"
                dated = True
            msgs.append({"role": role, "content": content})
    return msgs


def contains_judge(inst: LMEInstance, hypothesis: str) -> bool:
    """Deterministischer Offline-Judge: Referenz-Antwort (normalisiert) muss
    in der Hypothese vorkommen. Nur für Mechanik-Tests (--fake)."""
    if inst.is_abstention:
        return any(marker in hypothesis.lower()
                   for marker in ("kein", "nicht", "weiß nicht", "not", "no ",
                                  "unknown", "keine information"))
    return inst.answer.strip().lower() in hypothesis.lower()


def make_llm_judge(llm: LLMClient, judge_model: str,
                   api_base: str | None = None
                   ) -> Callable[[LMEInstance, str], Awaitable[bool]]:
    async def judge(inst: LMEInstance, hypothesis: str) -> bool:
        template = JUDGE_PROMPT_ABSTENTION if inst.is_abstention else JUDGE_PROMPT
        prompt = template.format(question=inst.question, answer=inst.answer,
                                 hypothesis=hypothesis)
        kwargs: dict[str, Any] = {"temperature": 0, "keep_alive": "30m"}
        if api_base:  # sonst landet litellm beim Default-Ollama (localhost)
            kwargs["api_base"] = api_base
        resp = await llm.complete(judge_model,
                                  [{"role": "user", "content": prompt}],
                                  **kwargs)
        verdict = (resp["choices"][0]["message"].get("content") or "").strip().lower()
        return verdict.startswith("yes")
    return judge


def _ingest_fingerprint(cfg: AppConfig, history: list[dict]) -> str:
    """Hash über alles, was das Ingestion-Ergebnis beeinflusst. Retrieval-/
    Antwort-Parameter gehören bewusst NICHT dazu — deren Experimente können
    die gecachte Ingestion wiederverwenden (coarse-to-fine-Iterationen)."""
    # temporal_supersede und user_profile ändern NUR das Prompt-Rendering
    # (query-seitig, auf der Kopie), nicht die gespeicherten Segmente/Tripel →
    # aus dem Ingestion-Fingerprint ausschließen, damit B/C den Cache nutzen.
    compression = {k: v for k, v in cfg.compression.model_dump().items()
                   if k not in ("temporal_supersede", "user_profile",
                                "entity_resolution")}
    relevant = {
        "sub_agents": cfg.sub_agents.model_dump(),
        "embedding": cfg.embedding.model_dump(),
        "chunk_size": cfg.storage.chunk_size,
        "compression": compression,
        "history": hashlib.md5(json.dumps(
            history, ensure_ascii=False).encode("utf-8")).hexdigest(),
    }
    return hashlib.md5(json.dumps(relevant, sort_keys=True,
                                  ensure_ascii=False,
                                  default=str).encode("utf-8")).hexdigest()


async def run_instance(cfg: AppConfig, llm: LLMClient, inst: LMEInstance,
                       judge: Callable[[LMEInstance, str], Awaitable[bool]],
                       workdir: Path) -> LMEResult:
    icfg = cfg.model_copy(deep=True)
    counter = TokenCounter(use_tiktoken=False)  # deterministisch, offline

    history = flatten_history(inst)

    # 1) Ingestion — gecacht pro Instanz + Kompressions-Config
    cache_db = workdir / f"{inst.question_id}.db"
    fp_file = workdir / f"{inst.question_id}.ingest.json"
    fingerprint = _ingest_fingerprint(icfg, history)

    t0 = time.perf_counter()
    meta: dict[str, Any] = {}
    if fp_file.exists() and cache_db.exists():
        meta = json.loads(fp_file.read_text(encoding="utf-8"))
    if meta.get("fingerprint") == fingerprint:
        sid = meta["session_id"]
        segments = int(meta.get("segments", 0))
        log.info("%s: Ingestion aus Cache (%d Segmente)",
                 inst.question_id, segments)
    else:
        cache_db.unlink(missing_ok=True)
        store = Store(cache_db)
        sid = store.create_session().id
        store.add_messages(sid, 0, history)
        workers = ExtractionWorkers(llm, icfg.sub_agents)
        pipeline = CompressionPipeline(store, workers, counter, icfg, llm=llm)
        segments = await pipeline.compress_session(sid)
        store.close()
        fp_file.write_text(json.dumps(
            {"fingerprint": fingerprint, "session_id": sid,
             "segments": segments}), encoding="utf-8")
    ingest_seconds = time.perf_counter() - t0

    # Frage-Phase läuft auf einer KOPIE, damit der Cache unverändert bleibt
    # (finalize_turn würde sonst Frage+Antwort in die Cache-DB schreiben).
    run_db = workdir / f"{inst.question_id}.run.db"
    shutil.copyfile(cache_db, run_db)
    icfg.storage.db_path = str(run_db)

    # Iteration B/C: Nutzerprofil und temporale Verdrängung wirken im
    # ARCHIVE-Prompt, der bei der Ingestion gebacken wurde. Für Cache-
    # Experimente (ohne Re-Ingestion) das Profil aus den gespeicherten
    # Summaries nachziehen und den Prompt neu rendern.
    if (icfg.compression.temporal_supersede or icfg.compression.user_profile
            or icfg.compression.entity_resolution):
        rstore = Store(str(run_db))
        rsession = rstore.get_session(sid)
        if rsession is not None:
            rpipe = CompressionPipeline(
                rstore, ExtractionWorkers(llm, icfg.sub_agents), counter, icfg)
            if icfg.compression.user_profile:
                await rpipe.build_and_store_profile(rsession.memory_id)
            new_prompt = rpipe.render_archive_prompt(rsession.memory_id)
            rstore.set_archive(sid, rsession.archived_upto, new_prompt)
        rstore.close()

    # 2) Frage als regulärer stateless Request (volle Historie + Frage)
    question = (f"(current date: {inst.question_date}) {inst.question}"
                if inst.question_date else inst.question)
    request_messages = history + [{"role": "user", "content": question}]

    from .runner import RecordingLLM

    rec = RecordingLLM(llm, icfg.main_llm.model)
    app = create_app(icfg, llm=rec)
    t1 = time.perf_counter()
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://eval",
                                     timeout=2400.0) as client:
            resp = await client.post(
                "/v1/chat/completions",
                headers={"x-session-id": sid},
                # num_ctx begrenzt den KV-Cache: gemma4 lädt sonst mit 262k
                # Kontext (Modelfile-Default) und wird zwischen Antwort und
                # Judge verdrängt → 1800-s-Reload. 40960 >> unser Upstream
                # (~30k). keep_alive hält das Modell resident (kein Thrashing
                # mit dem Judge-Modell). timeout als Sicherheitsnetz.
                json={"model": icfg.main_llm.model,
                      "messages": request_messages,
                      "num_ctx": 40960,
                      "keep_alive": "30m",
                      "timeout": 900},
            )
    finally:
        # Windows: SQLite-Datei muss geschlossen sein, bevor das Temp-Verzeichnis
        # aufgeräumt wird (ASGITransport fährt die Lifespan nicht).
        await app.state.compressor.drain()
        await app.state.compressor.stop()
        app.state.store.close()
        run_db.unlink(missing_ok=True)
    answer_seconds = time.perf_counter() - t1
    resp.raise_for_status()
    hypothesis = resp.json()["choices"][0]["message"].get("content") or ""

    upstream_tokens = (counter.count_messages(rec.main_calls[-1])
                       if rec.main_calls else 0)
    full_tokens = counter.count_messages(request_messages)
    hit = await judge(inst, hypothesis)

    return LMEResult(
        question_id=inst.question_id, question_type=inst.question_type,
        question=inst.question, expected=inst.answer, hypothesis=hypothesis,
        hit=hit, segments=segments, full_tokens=full_tokens,
        upstream_tokens=upstream_tokens, ingest_seconds=round(ingest_seconds, 2),
        answer_seconds=round(answer_seconds, 2),
    )


async def run_benchmark(cfg: AppConfig, llm: LLMClient,
                        instances: list[LMEInstance], total_in_dataset: int,
                        judge: Callable[[LMEInstance, str], Awaitable[bool]],
                        judge_description: str, dataset_name: str,
                        save_path: str | None = None,
                        prior: LMEReport | None = None,
                        workdir: str | None = None) -> LMEReport:
    """Fehler einzelner Instanzen brechen den Lauf nicht ab; nach jeder
    Instanz wird der Zwischenstand gespeichert (Resume über --json).
    workdir = persistenter Ingestion-Cache; ohne workdir Temp-Verzeichnis."""
    report = prior or LMEReport(dataset=dataset_name,
                                total_in_dataset=total_in_dataset,
                                judge_description=judge_description)
    tmp_ctx = tempfile.TemporaryDirectory(prefix="lme-") if workdir is None else None
    base = Path(tmp_ctx.name) if tmp_ctx else Path(workdir)
    base.mkdir(parents=True, exist_ok=True)
    try:
        for idx, inst in enumerate(instances, start=1):
            try:
                result = await run_instance(cfg, llm, inst, judge, base)
            except Exception as exc:  # Lauf fortsetzen, Fehler ausweisen
                log.error("[%d/%d] %s FEHLER: %s", idx, len(instances),
                          inst.question_id, exc)
                report.errors.append(f"{inst.question_id}: {exc}")
                continue
            report.results.append(result)
            log.info("[%d/%d] %s (%s): %s — %d Segmente, %d→%d Token, "
                     "ingest %.1fs, answer %.1fs",
                     idx, len(instances), inst.question_id, inst.question_type,
                     "HIT" if result.hit else "MISS", result.segments,
                     result.full_tokens, result.upstream_tokens,
                     result.ingest_seconds, result.answer_seconds)
            if save_path:
                with open(save_path, "w", encoding="utf-8") as fh:
                    json.dump(asdict(report), fh, ensure_ascii=False, indent=2)
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()
    return report


def load_prior_report(json_path: str) -> LMEReport | None:
    """Bestehende Ergebnisdatei -> Resume (bereits gelaufene IDs überspringen)."""
    path = Path(json_path)
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    report = LMEReport(dataset=raw.get("dataset", "?"),
                       total_in_dataset=raw.get("total_in_dataset", 0),
                       errors=list(raw.get("errors", [])),
                       judge_description=raw.get("judge_description", ""))
    report.results = [LMEResult(**r) for r in raw.get("results", [])]
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SemanticSwap LongMemEval (ADR-013)")
    parser.add_argument("--data", default="data/benchmarks/longmemeval_s.json")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--fake", action="store_true",
                        help="Offline-Mechanik-Modus (OfflineLLM + Substring-Judge)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--types", nargs="*", default=None)
    parser.add_argument("--judge-model", default=None,
                        help="Default: Summarization-Modell der Sub-Agenten")
    parser.add_argument("--json", dest="json_path")
    parser.add_argument("--workdir", default=None,
                        help="Persistenter Ingestion-Cache: Retrieval-"
                             "Experimente überspringen die teure Ingestion")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if hasattr(sys.stdout, "reconfigure"):  # Windows-Konsole (cp1252)
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    instances, total = load_instances(args.data, args.limit, args.types)
    if not instances:
        log.error("Keine Instanzen geladen (Filter zu streng?)")
        return 1

    if args.fake:
        cfg, llm = fake_config(), OfflineLLM()

        async def judge(inst: LMEInstance, hypothesis: str) -> bool:
            return contains_judge(inst, hypothesis)

        judge_description = "Substring-Judge (Offline-Mechanik, nicht publizierbar)"
    else:
        cfg = load_config(args.config)
        from ..llm import LiteLLMClient

        llm = LiteLLMClient()
        judge_model = args.judge_model or cfg.sub_agents.tasks.summarization
        judge = make_llm_judge(llm, judge_model,
                               api_base=cfg.sub_agents.api_base)
        judge_description = f"lokaler LLM-Judge ({judge_model})"

    prior = load_prior_report(args.json_path) if args.json_path else None
    if prior and prior.results:
        done = {r.question_id for r in prior.results}
        before = len(instances)
        instances = [i for i in instances if i.question_id not in done]
        log.warning("Resume: %d bereits gelaufene Instanzen übersprungen, "
                    "%d verbleiben", before - len(instances), len(instances))
        prior.errors = []  # fehlgeschlagene bekommen einen neuen Versuch

    report = asyncio.run(run_benchmark(
        cfg, llm, instances, total, judge, judge_description,
        dataset_name=Path(args.data).name, save_path=args.json_path,
        prior=prior, workdir=args.workdir))
    print(report.to_markdown())

    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as fh:
            json.dump(asdict(report), fh, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
