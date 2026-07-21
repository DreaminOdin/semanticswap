# LongMemEval-Experiment-Protokoll (M8, coarse-to-fine)

Strategie (PO 2026-07-19): erst breit streuen, dann fokussieren. Der
Ingestion-Cache (`--workdir data/benchmarks/lme_cache`) macht Retrieval-
Experimente billig (~40 min statt ~8 h für 24 Instanzen); nur Kompressions-
Änderungen (Profil, temporale Tripel, chunk_size) erfordern Re-Ingestion.

Standard-Kommando:
```
.venv\Scripts\python -m semanticswap.eval.longmemeval `
  --config <experiment.yaml> --data data\benchmarks\longmemeval_s_cleaned.json `
  --limit 24 --json data\benchmarks\<runN>.json --workdir data\benchmarks\lme_cache
```

| Run | Config | Änderung ggü. Baseline | Accuracy | Ersparnis | Status |
|---|---|---|---|---|---|
| 1 | config.gx10.yaml | Baseline (top_k 3, inj 2000) | **30,4 %** (7/23) | 8,01x | ✅ abgeschlossen |
| 2 | eval_config_gx10_run2.yaml | top_k 8, inj 6000 | **34,8 %** (8/23) | 5,36x | ✅ Budget hilft kaum; multi-session/preference bleiben 0 % → Module nötig |
| 3 | eval_config_gx10_run3.yaml | + hybrid (FTS5+RRF) | **54,2 %** (13/24) | 5,19x | ✅ **+19,4 pp!** alle Kategorien hoch, multi-session/preference 0→50 %; Cache-Lauf ~40 min |
| 4 | eval_config_gx10_run4.yaml | + Graph-Expansion (limit 3) | 68,2 % (n=22, 2 Timeouts) | 4,93x | ⚠️ **multi-session 50→0 %** trotz Zielsetzung; Gesamtzahl durch Timeouts verunreinigt → nicht sauber vergleichbar |
| 5 | eval_config_gx10_run5.yaml | Graph-Expansion limit 3→1 (Beifang-Test) | 54,5 % (n=22) | 5,16x | ❌ multi-session weiter **0/4** → Graph-Expansion **verworfen**; Hypothese widerlegt |
| 6 (geplant) | — | + Nutzer-Profil-Gedächtnis (Iteration B) | | | **Re-Ingestion nötig**; vorher GX10-Timeout-Fix |
| 7 (geplant) | — | + temporale Tripel (Iteration C) | | | **Re-Ingestion nötig** — mit B kombinieren |

| 6 | eval_config_gx10_run6.yaml | Hybrid bei KLEINEM Budget (top_k 3, inj 2000) | 50,0 % (n=22) | **8,14x** | ✅ **Produktions-Gewinner**: +19,6 pp vs Run 1 bei GEHALTENER Ersparnis |

| GX10-Fix | — | num_ctx 40k + keep_alive (Modell-Thrashing) | — | — | ✅ 24 Instanzen 0 Timeouts (vorher 2/24); saubere Daten |
| 7 | eval_config_gx10_run7.yaml | Gewinner + temporale Verdrängung (C) | 50,0 % (n=24, 0 Fehler) | 8,18x | ≈ neutral; knowledge-update 2/3→3/4 (Rauschen); Ersparnis gehalten |
| 8 | eval_config_gx10_run8.yaml | Gewinner + Profil-Gedächtnis (B) | (unterbrochen, n=17) | — | ⚠️ Lauf abgebrochen → sauber nachmessen |
| 9 | eval_config_gx10_run9.yaml | Gewinner + Re-Ranker (LLM-Listwise) | ~14 % (abgebr.) | — | ❌ **verworfen**: kleines Modell rankt nicht (nur "10,1"); Cross-Encoder nötig |
| 10 | eval_config_gx10_run10.yaml | Gewinner + temp. Verdrängung + Entity Resolution | 41,7 % (n=24) | 8,28x | ❌ Entity Resolution **schadet** (zu aggressive Kanonisierung) |
| 11 | eval_config_gx10_run11.yaml | Gewinner + Query-Decomposition | 37,5 % (n=24) | — | ❌ **schadet** (multi-session 0/4); kleines Modell zerlegt schlecht |
| 12 | eval_config_gx10_run12.yaml | Gewinner + **Cross-Encoder** (Kürzung 500) | 33,3 % (n=24) | 8,04x | ❌ schadet |
| 13 | (run12-Config, Kürzung 1800) | Cross-Encoder mit korrekter Kürzung | 29,2 % (n=24) | 7,95x | ❌ schadet weiter — Truncation war nicht die Ursache |

**ENDERGEBNIS der Serie (13 Läufe, 6 Techniken):** Nur die Hybrid-Suche gewinnt
(+20 pp, reproduzierbar). Alle Aufbauten — LLM-Re-Ranker, Cross-Encoder,
Graph-Expansion, Entity Resolution, Query-Decomposition — schaden; temporale
Verdrängung ist neutral. Ursache: faktischer Recall belohnt EXAKTEN Abgleich
(FTS), nicht semantische Ähnlichkeit. **Produktions-Config bleibt: nur Hybrid.**

**Gewinner-Konfiguration:** Run 6 — **Hybrid-Suche, top_k 3, inj 2000**:
**50,0 %** bei **8,14x** Ersparnis. Sauberster Vergleich (Run 1 vs 6, gleiches
Budget): Hybrid = +19,6 pp ohne Ersparnis-Verlust. Budget-Erhöhung (Run 3)
bringt nur +4 pp für −36 % Ersparnis → verworfen. Graph-Expansion verworfen.
→ **Produktions-Default: `retrieval.hybrid: true`, top_k 3, inj 2000.**

**Reihenfolge-Logik:** Retrieval-seitige Module (hybrid, graph) nutzen den
warmen Cache → billig, zuerst. Kompressions-seitige Module (Profil, temporale
Tripel) erzwingen Re-Ingestion → teuer, deshalb gebündelt in EINEN Lauf, erst
wenn die billige Retrieval-Konfiguration feststeht.

Die *Erkenntnisse* aus diesen Läufen (Trade-offs, welches Modul wofür, wie man
einstellt) werden im **[Tuning-Cookbook](tuning-cookbook.md)** verdichtet — das
ist die dauerhafte, auch für Anwender gedachte Lehre; diese Tabelle ist nur das
rohe Laufprotokoll.

Regeln:
- Pro Run genau EINE Änderung gegenüber dem Vorgänger-Run ausweisen.
- Gleiche 24 Instanzen (stratifiziert, Seed = Dataset-Reihenfolge) für
  Vergleichbarkeit; erst für publizierbare Zahlen auf 100+ gehen.
- Judge konstant halten (qwen2.5:7b lokal); jede Judge-Änderung wäre ein
  eigener Run.
- Ergebnisse hier eintragen + JSON unter data/benchmarks/ behalten.
