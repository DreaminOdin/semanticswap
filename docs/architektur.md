# SemanticSwap — Gesamtarchitektur

> Stand 2026-07-20. Lebendes Dokument; ergänzt die ADRs (verbindliche
> Einzelentscheidungen) um das große Bild. Messgrundlage der Bewertungen:
> [docs/eval/](eval/) (LongMemEval, Tuning-Cookbook).

## 1. Was SemanticSwap ist

Ein **OpenAI-kompatibler Inferenz-Proxy**, der zwischen Client (Chat-UI,
Coding-Agent, API-Client) und lokalem LLM (Ollama/LiteLLM) sitzt. Er
komprimiert lange Konversationen transparent in ein **semantisches
Langzeitgedächtnis** (SQLite: Segmente, Wissens-Graph, Vektoren, Volltext,
Profil) und spielt bei Bedarf die relevanten Original-Ausschnitte wieder ein.
Ziel: „Erinnern" weit über das Kontextfenster hinaus, bei drastisch weniger
Token — **lokal, ohne Cloud-Keys**.

Gemessen (LongMemEval-S, **98 Instanzen**, lokaler Judge, gemma4:26b):
**Kontext-Ersparnis ~8x, QA-Genauigkeit ~41 %** (Gewinner-Config Hybrid-Suche).
Kernpfad headless (ADR-006), GUI optional. (Eine frühere 24er-Stichprobe
zeigte optimistische 50 % — im Bestätigungslauf auf ~41 % korrigiert.)

## 2. Die zwei Pfade

Die Architektur hat einen **Schreibpfad** (Swap-Out, verdichten & ablegen) und
einen **Lesepfad** (Swap-In, gezielt wiederfinden). Beide teilen sich das
Gateway und den Speicher.

```
                          ┌─────────────── Client (OpenAI-API) ───────────────┐
                          │ request                                 response   │
                          ▼                                                    │
   ┌──────────────────────────────── Gateway ───────────────────────────────┐ │
   │ Auth v2 · Session-Tracker (Hash-Kette) · virtuelles Pruning             │ │
   └───────┬──────────────────────────────────────────────────┬─────────────┘ │
   Swap-In │ (Lesepfad)                          (Schreibpfad) │ Swap-Out       │
           ▼                                                    ▼               │
   ┌─────────────────┐                          ┌──────────────────────────┐   │
   │  Retrieval      │                          │  Context Monitor         │   │
   │  1 Hybrid-Suche │                          │  (Token-Schwelle)        │   │
   │    (Vektor+FTS  │                          └───────────┬──────────────┘   │
   │     +RRF)       │                                      ▼ (asynchron)      │
   │  2 Re-Ranker    │                          ┌──────────────────────────┐   │
   │    (LLM-listwise)│                         │  Job-Queue → Sub-Agenten │   │
   │  3 Budget       │                          │  Summary·Entity·Profil   │   │
   └───────┬─────────┘                          └───────────┬──────────────┘   │
           │  Original-Snippets                             │ Segmente/Tripel/  │
           │                                                │ Embeddings/Profil │
           ▼                                                ▼                   │
   ┌────────────────────────── Semantic Memory (SQLite) ─────────────────────┐ │
   │ segments · triples (Graph) · embeddings · segments_fts · profiles       │ │
   └─────────────────────────────────────────────────────────────────────────┘ │
           │                                                                    │
           ▼ (injizierte Erinnerung + ARCHIVE-Prompt)                           │
   ┌──────────────────────── Haupt-LLM (Ollama/LiteLLM) ────────────────────────┘
   └─────────────────────────────────────────────────────────────────────────►
```

## 3. Schreibpfad (Swap-Out, Modus B — Post-Response Batch)

Nach jeder Antwort prüft der **Context Monitor** die Token-Füllung. Über der
Schwelle stößt er die Kompression **asynchron** an (blockiert die Antwort nie):

1. **Segmentierung** — der Verlauf wird in Segmente geschnitten
   (aktuell feste Chunk-Größe; *geplant: semantisches Chunking*, §6).
2. **Map — Sub-Agenten pro Segment** (kleine Modelle, parallel, gedeckelt):
   - *Summary-Worker*: verdichtet + vergibt Priorität high/low (ADR-011).
   - *Entity-Worker*: extrahiert Wissens-Tripel (Subjekt·Prädikat·Objekt).
   - *Profil-Worker* (Iteration B): destilliert ein stehendes Nutzerprofil.
3. **Reduce** — konsolidierter **ARCHIVE-Prompt** über alle Segmente:
   Summaries (mit Deep-Archive-GC für alte Low-Prio), Graph-Tripel
   (optional temporale Verdrängung, Iteration C), Nutzerprofil oben.
4. **Persistenz** — Segmente, Tripel, Embeddings, FTS-Index, Profil in SQLite.
5. **Virtuelles Pruning** (ADR-003): Der Client-Verlauf wird nie verändert.
   Beim nächsten Request ersetzt das Gateway den archivierten Prefix durch den
   kompakten ARCHIVE-Prompt — daher die Token-Ersparnis.

## 4. Lesepfad (Swap-In)

Bei jedem Request kann relevantes Original-Material temporär zurückgeholt werden:

1. **Hybrid-Suche** (Gewinner, +19,6 pp gemessen): Vektor-Ähnlichkeit **und**
   FTS5-Volltext (exakte Namen/Zahlen/Codes), verschmolzen per Reciprocal Rank
   Fusion. Fällt das Embedding aus, degradiert es zur reinen Volltextsuche.
2. **Re-Ranker** (Prio 1, LLM-listwise): bewertet Query+Kandidat gemeinsam und
   sortiert neu — gegen das Beifang-Problem. *(in Messung)*
3. **Budget** — Original-Snippets bis zum Token-Limit als temporäre
   System-Message injiziert (nie persistiert).

Dazu **Active Memory Retrieval** (ADR-010): Bei Requests ohne eigene Tools
erhält das Haupt-LLM ein `retrieve_archived_memory`-Tool und kann selbst
nachladen; der Tool-Zyklus ist für den Client unsichtbar.

## 5. Querschnitt

- **Auth v2** (ADR-014): Tailnet-Vertrauen (passwortlos für Betreiber),
  Geräte-Cookies + Rollen (family/admin) für Familie, Login-Bremse,
  Zugriffs-Logbuch. Jede Prüfung auf dem Host (Funnel umgeht die Edge).
- **Observability** (ADR-012): In-Process-Event-Bus → SSE → Live-Flowchart.
  Jede Pipeline-Stufe emittiert typisierte Events (`retrieval_search`,
  `rerank`, `subagent_*`, `compression_*`, `access_denied` …).
- **Storage** (ADR-004): SQLite, Vektor-Store hinter Interface
  (aktuell Brute-Force-Cosine; *geplant: ANN-Index*, §6).
- **Forks** (ADR-009): teilen den semantischen Speicher des Ursprungs
  (`memory_id`).

## 6. Bewusste Lücken & Roadmap (nach Impact)

Aus dem Architektur-Vergleich (Zep/Graphiti, Letta/MemGPT, mem0, Standard-RAG)
und den eigenen Messungen abgeleitet. Der rote Faden: **Retrieval-Präzision
und Entitäts-Qualität** sind der Hebel, nicht weiteres Speicher-Schema.

| # | Baustein | Warum | Status |
|---|---|---|---|
| 1 | **Re-Ranker** LLM-Listwise | größter erwarteter Präzisionsgewinn | **gebaut, gemessen — VERWORFEN**: kleines Modell liefert kein brauchbares Ranking (14 %). Echter **Cross-Encoder** (bge-reranker) = richtiger Weg, braucht Abhängigkeit → PO-Entscheidung |
| 2 | **Entity Resolution** (Kanonisierung) | repariert Graph + temporale Verdrängung | gebaut (deterministisch), in Messung |
| 3 | **Query-Decomposition** | gegen multi-session (Multi-Hop) | gebaut, Messung offen |
| 4 | **Semantisches Chunking** | Topic-Grenzen statt nur Größe | Segmenter respektiert bereits Message-Grenzen; Embedding-basierte Topic-Grenzen offen (braucht Re-Ingestion) |
| 5 | **Vektorsuche** | Skalierung | numpy-vektorisiert (~100x, keine Abhängigkeit); echter ANN-Index (HNSW/sqlite-vec) offen → Abhängigkeitsentscheidung |
| 6 | **Selbst-editierendes Gedächtnis** (Letta-Stil) | Modell kuratiert eigenes Gedächtnis | Fernwette |

**Verworfen (mit Beleg):** (a) Graph-Expansion über rohe Entitäten
(verschlechterte multi-session); (b) LLM-Listwise-Re-Ranker (kleines Modell
kann nicht ranken). Beides im [Tuning-Cookbook](eval/tuning-cookbook.md)
belegt.

**Roter Faden aus 13 Messläufen (final):** Nur die **Hybrid-Suche** gewinnt
(+20 pp, reproduzierbar). **Jede** semantische Aufbau-Stufe — LLM-Re-Ranker,
**Cross-Encoder** (das „richtige" Werkzeug!), Graph-Expansion, Entity
Resolution, Query-Decomposition — hat auf dem Benchmark **geschadet**; temporale
Verdrängung war neutral. Die Erkenntnis ist tiefer als „kleines Modell":
**Faktischer Recall aus komprimiertem Gedächtnis belohnt exakten Abgleich
(FTS/BM25), nicht semantische Ähnlichkeit.** Re-Ranker & Co. optimieren auf
Ähnlichkeit und verdrängen dabei die Passage mit dem konkreten Fakt. Deshalb ist
Hybrid der Gewinn (es *fügt* exaktes Matching hinzu) und deshalb schaden die
Aufbauten. Beleg: [Tuning-Cookbook](eval/tuning-cookbook.md), §„exakter Treffer
schlägt semantische Ähnlichkeit". **Konsequenz: Architektur schlank halten —
Hybrid ist die Produktions-Config; die Aufbau-Module bleiben als abschaltbare
Experimente im Code (default aus).** Größere Gewinne bräuchten ein stärkeres
Antwort-Modell oder n=100+, nicht mehr Retrieval-Schichten.

## 7. Warum diese Architektur (Abgrenzung)

- Gegen **Anthropic Context-Editing**: OpenAI-kompatibel + lokal + komprimiert-
  und-stellt-wieder-her statt nur zu verwerfen.
- Gegen **mem0/Zep-SaaS**: local-by-design, Graph gratis eingebaut (nicht hinter
  Paywall), eine SQLite-Datei statt externer DB.
- Gegen **Letta**: kein Agent-Runtime-Lock-in — reiner Proxy, jeder OpenAI-Client
  bleibt austauschbar.
