"""M5 Eval-Runner: fährt das Szenario durch den Proxy und misst die Zielmetriken.

Metriken (Zielwerte laut PO-Entscheidung 2026-07-15):
- Archiv-Kompressions-Ratio (Original-Token / ARCHIVE-Prompt-Token), Ziel >= 5:1
- Recall-Quote über die Referenzcode-Fragen, Ziel >= 80 %
- Proxy-Overhead pro Turn (Proxy-Latenz minus direkte LLM-Latenz), Ziel < 100 ms
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import httpx

from ..config import AppConfig
from ..gateway import create_app
from ..llm import LLMClient
from ..tokens import TokenCounter
from .scenario import Question

TARGET_ARCHIVE_RATIO = 5.0
TARGET_RECALL = 0.8
TARGET_OVERHEAD_MS = 100.0


class RecordingLLM:
    """Wrapper, der Upstream-Messages und die im Haupt-Modell verbrachte Zeit
    protokolliert. Der Proxy-Overhead ist dann exakt: Gesamtlatenz des
    Proxy-Requests minus reine Modell-Zeit (Retrieval-Embedding, Session-
    Tracking und Pruning zählen korrekt als Overhead)."""

    def __init__(self, inner: LLMClient, main_model: str):
        self.inner = inner
        self.main_model = main_model
        self.main_calls: list[list[dict]] = []
        self.main_durations_ms: list[float] = []

    async def complete(self, model: str, messages: list[dict], **kwargs: Any) -> dict:
        if model == self.main_model:
            self.main_calls.append(messages)
            t0 = time.perf_counter()
            result = await self.inner.complete(model, messages, **kwargs)
            self.main_durations_ms.append((time.perf_counter() - t0) * 1000)
            return result
        return await self.inner.complete(model, messages, **kwargs)

    def complete_stream(self, model: str, messages: list[dict],
                        **kwargs: Any) -> AsyncIterator[dict]:
        if model == self.main_model:
            self.main_calls.append(messages)
        return self.inner.complete_stream(model, messages, **kwargs)

    async def embed(self, model: str, texts: list[str],
                    **kwargs: Any) -> list[list[float]]:
        return await self.inner.embed(model, texts, **kwargs)


@dataclass
class QuestionResult:
    question: str
    expected: str
    answer: str
    hit: bool


@dataclass
class EvalReport:
    turns: int
    session_id: str
    archived_upto: int
    segments: int
    triples: int
    archive_ratio: float | None
    context_ratio: float | None
    client_tokens_final: int
    upstream_tokens_final: int
    proxy_overhead_ms: float
    recall_rate: float
    question_results: list[QuestionResult] = field(default_factory=list)

    def to_markdown(self, fake_mode: bool = False) -> str:
        def status(ok: bool | None) -> str:
            return "-" if ok is None else ("PASS" if ok else "FAIL")

        ratio = f"{self.archive_ratio:.1f}:1" if self.archive_ratio else "n/a"
        ctx = f"{self.context_ratio:.2f}x" if self.context_ratio else "n/a"
        lines = [
            "# SemanticSwap Eval-Report" + (" (Offline-Mechanik-Modus)" if fake_mode else ""),
            "",
            f"Turns: {self.turns} · Session: {self.session_id} · "
            f"archiviert bis Message {self.archived_upto} · "
            f"{self.segments} Segmente · {self.triples} Tripel",
            "",
            "| Metrik | Wert | Ziel | Status |",
            "|--------|------|------|--------|",
            f"| Archiv-Kompressions-Ratio | {ratio} | >= {TARGET_ARCHIVE_RATIO:.0f}:1 | "
            f"{status(self.archive_ratio >= TARGET_ARCHIVE_RATIO if self.archive_ratio else None)} |",
            f"| Recall-Quote | {self.recall_rate:.0%} | >= {TARGET_RECALL:.0%} | "
            f"{status(self.recall_rate >= TARGET_RECALL)} |",
            f"| Proxy-Overhead (Median/Turn) | {self.proxy_overhead_ms:.1f} ms | "
            f"< {TARGET_OVERHEAD_MS:.0f} ms | "
            f"{status(self.proxy_overhead_ms < TARGET_OVERHEAD_MS)} |",
            f"| Kontext-Ersparnis (Client vs. Upstream, letzte Frage) | {ctx} | - | - |",
            "",
            f"Token letzte Frage: Client {self.client_tokens_final} -> "
            f"Upstream {self.upstream_tokens_final}",
            "",
            "## Fragen",
        ]
        for qr in self.question_results:
            mark = "[OK]  " if qr.hit else "[MISS]"
            lines.append(f"- {mark} {qr.question} -> erwartet `{qr.expected}`, "
                         f"Antwort: {qr.answer}")
        if fake_mode:
            lines.append("")
            lines.append("_Offline-Modus: misst die MECHANIK des Speichersystems "
                         "(extraktives Offline-LLM), nicht die Qualität echter Modelle._")
        return "\n".join(lines)

    def targets_met(self) -> bool:
        return (
            (self.archive_ratio or 0) >= TARGET_ARCHIVE_RATIO
            and self.recall_rate >= TARGET_RECALL
            and self.proxy_overhead_ms < TARGET_OVERHEAD_MS
        )


async def run_eval(cfg: AppConfig, llm: LLMClient, turns: list[str],
                   questions: list[Question]) -> EvalReport:
    counter = TokenCounter(use_tiktoken=False)
    recording = RecordingLLM(llm, cfg.main_llm.model)
    app = create_app(cfg, llm=recording)
    history: list[dict] = [{
        "role": "system",
        "content": "Du bist ein hilfreicher Assistent mit Langzeitgedächtnis.",
    }]
    overheads: list[float] = []
    session_id = ""

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://eval",
                                 timeout=600) as client:
        for text in turns:
            history.append({"role": "user", "content": text})
            t0 = time.perf_counter()
            resp = await client.post("/v1/chat/completions",
                                     json={"model": "eval", "messages": history})
            proxy_ms = (time.perf_counter() - t0) * 1000
            resp.raise_for_status()
            session_id = resp.headers["x-semanticswap-session"]
            answer = resp.json()["choices"][0]["message"]["content"]

            model_ms = (recording.main_durations_ms[-1]
                        if recording.main_durations_ms else 0.0)
            overheads.append(max(0.0, proxy_ms - model_ms))
            history.append({"role": "assistant", "content": answer})

        await app.state.compressor.drain()

        results: list[QuestionResult] = []
        for q in questions:
            msgs = history + [{"role": "user", "content": q.text}]
            resp = await client.post("/v1/chat/completions",
                                     json={"model": "eval", "messages": msgs})
            resp.raise_for_status()
            answer = resp.json()["choices"][0]["message"]["content"]
            results.append(QuestionResult(
                question=q.text, expected=q.expected, answer=answer,
                hit=q.expected.lower() in answer.lower(),
            ))
        await app.state.compressor.drain()

        store = app.state.store
        session = store.get_session(session_id)
        archive_ratio = None
        if session and session.archived_upto > 0 and session.archive_prompt:
            archived = store.get_messages(session_id)[:session.archived_upto]
            original_tokens = counter.count_messages([m.raw for m in archived])
            archive_tokens = counter.count_text(session.archive_prompt)
            if archive_tokens:
                archive_ratio = original_tokens / archive_tokens

        client_tokens = counter.count_messages(
            history + [{"role": "user", "content": questions[-1].text}])
        upstream_tokens = counter.count_messages(recording.main_calls[-1])
        context_ratio = client_tokens / upstream_tokens if upstream_tokens else None
        stats = store.stats()

    hits = sum(1 for r in results if r.hit)
    return EvalReport(
        turns=len(turns),
        session_id=session_id,
        archived_upto=session.archived_upto if session else 0,
        segments=stats["segments"],
        triples=stats["triples"],
        archive_ratio=archive_ratio,
        context_ratio=context_ratio,
        client_tokens_final=client_tokens,
        upstream_tokens_final=upstream_tokens,
        proxy_overhead_ms=statistics.median(overheads) if overheads else 0.0,
        recall_rate=hits / len(results) if results else 0.0,
        question_results=results,
    )
