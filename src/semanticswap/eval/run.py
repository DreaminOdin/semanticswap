"""CLI für den M5-Eval.

Offline (Mechanik-Smoke, kein Netzwerk):
    python -m semanticswap.eval.run --fake

Gegen echte Modelle (lokales Ollama gemäß config.yaml, keine Cloud-Keys nötig):
    python -m semanticswap.eval.run --config config.yaml [--topics 8] [--filler 10]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys

from ..config import AppConfig, load_config
from .fakes import OfflineLLM
from .runner import run_eval
from .scenario import build_scenario


def fake_config() -> AppConfig:
    return AppConfig.model_validate({
        "main_llm": {"provider": "offline", "model": "offline/main",
                     "max_context_tokens": 400, "trigger_thresholds": [0.5]},
        "sub_agents": {"provider": "offline", "concurrency_limit": 4,
                       "processing_mode": "batch",
                       "tasks": {"summarization": "offline/summary",
                                 "entity_extraction": "offline/entity",
                                 "synthesizer": "offline/synth"}},
        "embedding": {"model": "offline/embed", "enabled": True},
        "storage": {"db_path": ":memory:", "chunk_size": 300},
        "compression": {"keep_recent_messages": 2},
        "retrieval": {"enabled": True, "top_k": 2, "min_score": 0.05,
                      "max_injection_tokens": 300},
    })


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SemanticSwap M5 Eval")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--fake", action="store_true",
                        help="Offline-Mechanik-Modus (deterministisch, kein LLM nötig)")
    parser.add_argument("--topics", type=int, default=6)
    parser.add_argument("--filler", type=int, default=10)
    parser.add_argument("--json", dest="json_path",
                        help="Report zusätzlich als JSON speichern")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):  # Windows-Konsole (cp1252)
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if args.fake:
        cfg, llm = fake_config(), OfflineLLM()
    else:
        cfg = load_config(args.config)
        from ..llm import LiteLLMClient

        llm = LiteLLMClient()

    turns, questions = build_scenario(args.topics, args.filler)
    report = asyncio.run(run_eval(cfg, llm, turns, questions))
    print(report.to_markdown(fake_mode=args.fake))

    if args.json_path:
        from dataclasses import asdict

        with open(args.json_path, "w", encoding="utf-8") as fh:
            json.dump(asdict(report), fh, ensure_ascii=False, indent=2)

    # Zielwerte sind nur im Real-Modus verbindlich (PO-Entscheidung M5)
    if not args.fake and not report.targets_met():
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
