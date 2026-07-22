# LongMemEval-Experiment-Protokoll

Rohes Laufprotokoll aller Messungen. Die *Erkenntnisse* daraus (Trade-offs,
welches Modul wofür, wie man einstellt) sind im
**[Tuning-Cookbook](tuning-cookbook.md)** verdichtet — das ist die dauerhafte,
auch für Anwender gedachte Lehre.

**Setup:** LongMemEval-S (500 Instanzen à ~115k Token), stratifizierte
Stichprobe von 24, Antwortmodell gemma4:26b, Worker/Judge qwen2.5:7b, alles
lokal via Ollama. Judge ist NICHT der GPT-4o-Judge des Papers → Zahlen sind
**self-reported**.

**Ingestion-Cache** (`--workdir`) macht Retrieval-Experimente billig
(~40 min statt ~8 h je 24 Instanzen). Kompressions-Module wirken im
ARCHIVE-Prompt und werden query-seitig neu gerendert — dadurch ebenfalls
cache-testbar.

```
.venv\Scripts\python -m semanticswap.eval.longmemeval `
  --config <experiment.yaml> --data data\benchmarks\longmemeval_s_cleaned.json `
  --limit 24 --json data\benchmarks\<runN>.json --workdir data\benchmarks\lme_cache
```

## Läufe

| # | Änderung ggü. Vorgänger | Accuracy | Ersparnis | Befund |
|---|---|---|---|---|
| 1 | Baseline (top_k 3, inj 2000) | 30,4 % (n=23) | 8,01x | Ausgangspunkt |
| 2 | Budget: top_k 8, inj 6000 | 34,8 % (n=23) | 5,36x | +4 pp für −33 % Ersparnis → schlechter Tausch |
| 3 | + Hybrid-Suche (FTS5+RRF) | 54,2 % (n=24) | 5,19x | **+19,4 pp**, alle Kategorien hoch |
| 4 | + Graph-Expansion (limit 3) | 68,2 % (n=22, 2 Timeouts) | 4,93x | ⚠️ Zielkategorie multi-session 50→**0 %**; Gesamtzahl durch Timeouts verunreinigt |
| 5 | Graph-Expansion limit 3→1 | 54,5 % (n=22) | 5,16x | ❌ multi-session weiter 0/4 → **Graph-Expansion verworfen**, Beifang-Hypothese widerlegt |
| 6 | **Hybrid bei kleinem Budget** (top_k 3, inj 2000) | **50,0 %** (n=22) | **8,14x** | ✅ **Produktions-Gewinner**: +19,6 pp vs. Run 1 bei gehaltener Ersparnis |
| — | *Infra-Fix:* num_ctx 40k + keep_alive | — | — | Modell-Thrashing beseitigt: 24 Instanzen, 0 Timeouts (vorher 2/24) |
| 7 | + temporale Verdrängung | 50,0 % (n=24) | 8,18x | ≈ neutral |
| 8 | + Profil-Gedächtnis | *(Lauf unterbrochen, n=17)* | — | verworfen, siehe Run 14 |
| 9 | + Re-Ranker (LLM-Listwise) | ~14 % (abgebrochen) | — | ❌ kleines Modell rankt nicht (gab nur „10, 1") |
| 10 | + Entity Resolution | 41,7 % (n=24) | 8,28x | ❌ schadet — zu aggressive Kanonisierung |
| 11 | + Query-Decomposition | 37,5 % (n=24) | — | ❌ schadet, multi-session 0/4 |
| 12 | + **Cross-Encoder** (Kürzung 500) | 33,3 % (n=24) | 8,04x | ❌ schadet |
| 13 | Cross-Encoder, Kürzung 1800 | 29,2 % (n=24) | 7,95x | ❌ Truncation war nicht die Ursache |
| 14 | + Profil-Gedächtnis (sauber) | 43,5 % (n=23) | 8,00x | ⚠️ netto negativ, **aber Zielkategorie `preference` 75 %** — bester Wert aller Läufe |
| **✅ Bestätigung** | **Gewinner-Config auf 98 Instanzen** | **40,8 %** (40/98) | **8,07x** | belastbare Launch-Zahl: die 24er-Stichprobe (50 %) war optimistisch |

## Ergebnis

**Nur die Hybrid-Suche gewinnt** (+20 pp, reproduzierbar, ohne Ersparnis-
Verlust). Jede semantische Aufbaustufe — LLM-Re-Ranker, Cross-Encoder,
Graph-Expansion, Entity Resolution, Query-Decomposition — **schadet**;
temporale Verdrängung ist neutral; das Profil-Gedächtnis hilft nur seiner
Zielkategorie und kostet netto.

**Ursache:** Faktischer Recall belohnt **exakten Abgleich** (FTS/BM25), nicht
semantische Ähnlichkeit — Re-Ranker & Co. verdrängen die Passage mit dem
konkreten Fakt. Ausführlich im [Cookbook](tuning-cookbook.md).

→ **Produktions-Default: `retrieval.hybrid: true`, top_k 3, inj 2000.**

## Methodik-Regeln

- Pro Lauf genau EINE Änderung gegenüber dem Vorgänger ausweisen.
- Vergleiche bei konstanter Nebenvariable (sonst wird der Effekt falsch
  zugeschrieben — passiert bei Run 2→3, korrigiert durch Run 1 vs. 6).
- Läufe mit unterschiedlichem n nie direkt vergleichen.
- Judge konstant halten; jede Judge-Änderung wäre ein eigener Lauf.
- Übersprungene Instanzen (Timeout/Fehler) ausweisen, nie stillschweigend
  aus der Quote nehmen.
- Kategorie-Werte bei n=4 sind verrauscht — nur Muster über mehrere Läufe
  interpretieren.
