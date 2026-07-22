# SemanticSwap — Memory-Optimized Inference Proxy

OpenAI-kompatibler Proxy, der das Kontextfenster von LLMs dynamisch komprimiert,
strukturiert in einen semantischen Speicher (SQLite: Graph-Tripel + Vektoren)
auslagert und bei Bedarf wieder hereinlädt. Basis: PAD "SemanticSwap",
Projektplan: [../PROJEKTENTWURF.md](../PROJEKTENTWURF.md).

## Status

Produktiv einsetzbar. Gateway mit Session-Tracking (Prefix-Hash-Kette),
Swap-Out (Modus B) mit virtuellem Pruning, Swap-In mit **Hybrid-Retrieval**
(Volltext + Vektoren, RRF), Active Memory Retrieval, semantischer Garbage
Collector, Auth (Geräte-Cookies/Rollen/Zugriffs-Logbuch) und Live-GUI.

**Gemessen** (LongMemEval-S, **98 Instanzen**, lokaler Judge, gemma4:26b):
**~8x Kontext-Ersparnis bei ~41 % QA-Genauigkeit.** Die Hybrid-Suche
(Volltext + Vektoren) war der große Hebel (+20 pp gegenüber reiner
Vektorsuche); jede weitere semantische Aufbaustufe hat auf dem Benchmark
geschadet. Vollständige Messreihe (14 Läufe, 6 Techniken), Trade-offs und
Lehren: **[docs/eval/tuning-cookbook.md](docs/eval/tuning-cookbook.md)**.
Gesamtarchitektur: [docs/architektur.md](docs/architektur.md) ·
Entscheidungen: [docs/adr/](docs/adr/).

> **Ehrliche Einordnung:** Die Zahlen sind self-reported — lokaler Judge
> (qwen2.5:7b) statt GPT-4o wie im Paper, und ein 26B-Antwortmodell. Sie sind
> daher nicht direkt mit Paper-Bestwerten vergleichbar; das Kernversprechen ist
> die **verlustarme 8x-Kompression bei vollständig lokalem Betrieb**. Eine
> frühere 24er-Stichprobe zeigte optimistische 50 % — der 98er-Lauf korrigierte
> das auf ~41 % (kleine Stichproben täuschen; genau dafür der Bestätigungslauf).

Alle Inferenz läuft lokal über **Ollama** — wahlweise auf dieser Maschine (Default)
oder remote (z. B. GX10) per `api_base` in der [config.yaml](config.yaml).

## Setup

```powershell
python -m venv .venv
.venv\Scripts\pip install -e ".[dev]"
```

## Starten (headless)

```powershell
.venv\Scripts\python -m semanticswap.main config.yaml
```

Der Proxy lauscht auf `http://127.0.0.1:8080/v1/chat/completions` und ist für
jeden OpenAI-Client ein Drop-in-Replacement (Base-URL umbiegen, fertig).
Optional: Header `x-session-id` für explizite Session-Zuordnung.

Admin-API (read-only): `/admin/stats`, `/admin/sessions`, `/admin/sessions/{id}`,
`/health`. Optionale **Web-GUI**: `http://127.0.0.1:8080/ui` (Dashboard,
Session-Details mit ARCHIVE-Prompt, Segmenten und Graph-Tripeln) und
`http://127.0.0.1:8080/ui/flow` — **Echtzeit-Flowchart**: Architektur-Knoten
leuchten auf, wenn die Komponente arbeitet (z. B. Sub-Agenten im Hintergrund),
mit Live-Ereignisprotokoll (Server-Sent-Events, ADR-012). Seit M8 mit eigenem
**Hybrid-Retrieval-Knoten**: Jede Speicher-Suche (Vektor + FTS5-Volltext,
RRF-Fusion) erscheint als `retrieval_search`-Event mit Trefferzahlen beider
Suchpfade; Hover-Tooltips erklären jeden Knoten.

**Active Memory Retrieval** (ADR-010): Bei Requests ohne eigene Tools erhält das
Haupt-LLM das Tool `retrieve_archived_memory` und kann archivierte Original-
Transkripte selbst nachladen; der Tool-Zyklus ist für den Client unsichtbar.

## Tests

```powershell
.venv\Scripts\python -m pytest
```

Unit- und Integrationstests laufen komplett offline gegen ein deterministisches
FakeLLM (ADR-007).

## Eval (M5)

```powershell
# Offline-Mechanik-Messung (deterministisch, kein Modell nötig):
.venv\Scripts\python -m semanticswap.eval.run --fake

# Echte Messung gegen lokales Ollama (Modelle aus config.yaml):
.venv\Scripts\python -m semanticswap.eval.run --config config.yaml
```

Misst Archiv-Kompressions-Ratio (Ziel ≥ 5:1), Recall-Quote (Ziel ≥ 80 %) und
Proxy-Overhead (Ziel < 100 ms) über eine synthetische lange Konversation mit
verankerten Fakten.

## Lizenz

**Source-available, nicht Open Source im OSI-Sinn:**
[PolyForm Noncommercial 1.0.0](LICENSE.md) — kostenlos für Forschung, Lehre,
Non-Profits, Behörden und Privatnutzung. Kommerzielle Nutzung erfordert eine
[kommerzielle Lizenz](COMMERCIAL-LICENSE.md) (PO-Entscheidung 2026-07-18).
Externe Beiträge nur nach Contributor-Vereinbarung (CLA), damit die
Doppellizenzierung möglich bleibt.

## Konfiguration

Siehe [config.yaml](config.yaml): Haupt-LLM + Schwellenwerte (`main_llm`),
Sub-Agenten-Modelle und Parallelität (`sub_agents`), Embeddings (`embedding`),
Speicher (`storage`), Kompressionsverhalten (`compression`).
Modellnamen in LiteLLM-Notation, z. B. `gpt-4o` oder `ollama/llama3:8b-instruct`.
