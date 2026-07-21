# ADR-013: Externe Benchmarks (LongMemEval zuerst)

**Status:** Akzeptiert · **Datum:** 2026-07-18

## Kontext
Marktrecherche (2026-07-18, 103-Agenten-Deep-Research): Der Memory-Layer-Markt
ist dicht besetzt; Glaubwürdigkeit entsteht über **öffentliche, unabhängige
Benchmarks** — dort sind die Wettbewerber messbar schwach (Mem0: 3,4 % Conflict
Resolution auf MemoryAgentBench). Unser internes „Recall 100 %" (M5-Szenario)
ist extern nicht vergleichbar. Ziel: publizierbare Zahlen, komplett lokal
erhoben (PO-Regel: keine Cloud-Keys).

## Entscheidung
1. **LongMemEval zuerst** (ICLR 2025, MIT-Lizenz, 500 Instanzen à ~115k Token,
   5 Fähigkeits-Typen). **LoCoMo wird NICHT verwendet:** CC BY-NC 4.0 ist mit
   der Kaufprodukt-Regel (nur MIT/Apache/BSD-artige Fremdinhalte) unvereinbar.
   MemoryAgentBench folgt ggf. später (Lizenz vor Nutzung prüfen).
2. **Ingestion direkt über Store + CompressionPipeline** (`compress_session`),
   nicht Turn-für-Turn durchs Gateway: Die Haystack-Historie ist im Datensatz
   vorgegeben; sie pro Turn von einem LLM „beantworten" zu lassen wäre
   Rechenzeit ohne Erkenntnis. Die **Frage** läuft dagegen durch den echten
   Request-Pfad (Gateway mit explizieter Session-ID → virtuelles Pruning,
   ARCHIVE-Prompt, Swap-In, Active Retrieval) — gemessen wird das
   Gedächtnis-Subsystem end-to-end.
3. **Judge = lokales LLM** (konfigurierbar, Default Worker-Modell) mit dem
   Ja/Nein-Protokoll des Papers. Abweichung vom Original (GPT-4o-Judge) wird
   in jedem Report ausgewiesen; Zahlen sind damit „self-reported, local
   judge" — ehrlich labeln, nie als identische Methodik verkaufen.
4. **Keine stillen Caps:** Teil-Läufe (`--limit`, `--types`) loggen sichtbar,
   was ausgelassen wurde; Reports nennen Stichprobengröße pro Typ.
5. Datensatz wird **nicht eingecheckt** (fremdes Werk, ~Dutzende MB); Bezug
   dokumentiert (Hugging Face `xiaowu0162/longmemeval-cleaned`), Ablage lokal
   unter `data/benchmarks/`.

## Konsequenzen
- Vergleichbare Kernmetrik (QA-Accuracy gesamt + pro Fähigkeits-Typ) plus
  unsere Zusatzmetriken (Upstream-Token vs. Voll-Kontext = Ersparnis).
- Judge-Abweichung begrenzt die direkte Vergleichbarkeit mit Paper-Zahlen —
  akzeptiert, da Cloud-Judges gegen die PO-Regel verstoßen würden.
- Laufzeit auf dem GX10 ist der Engpass → Pilot mit stratifizierter
  Stichprobe, voller Lauf als Nacht-Job (Lastregeln: Parallelität 1–2,
  kleine Worker-Modelle).
