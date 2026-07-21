# LongMemEval-Pilot #1 (2026-07-19) — interne Baseline, NICHT publizieren

**Setup:** `longmemeval_s_cleaned` (500 Instanzen à ~115k Token), stratifizierte
Stichprobe **24 Instanzen** (4 je Typ; 1 Timeout-Ausfall → n=23). Antwortmodell
gemma4:26b, Worker qwen2.5:7b, Judge qwen2.5:7b (Gegenprobe gemma4:26b:
identische 30,4 %, 2 Urteile netto-neutral gekippt → Judge ist nicht der
Engpass). Konfiguration = Produktions-Defaults (`config.gx10.yaml`:
top_k 3, max_injection_tokens 2000, chunk_size 1000). Läufe auf GX10
(Ollama via SSH-Tunnel), Runner: `python -m semanticswap.eval.longmemeval`.

## Ergebnis

- **QA-Accuracy: 30,4 %** (7/23) — nicht launch-tauglich
- **Kontext-Ersparnis: 8,01x** (Ø ~124k → ~15,5k Token) — launch-tauglich
- Ingestion Ø ~20 min/Instanz, Antwort Ø ~80 s (Ausreißer bei gemma-Reload:
  Ollama lädt gemma4:26b mit 262k-Kontext, Reload kann Minuten dauern; ein
  Call lief in den 1800-s-Timeout)

| Typ | Accuracy | Interpretation |
|---|---|---|
| single-session-user | 75 % (3/4) | Einzelfakten aus User-Turns: funktioniert |
| single-session-assistant | 50 % (2/4) | Assistant-Aussagen werden schwächer erinnert |
| knowledge-update | 25 % (1/4) | Updates überschreiben alte Fakten nicht zuverlässig |
| temporal-reasoning | 25 % (1/4) | Sitzungsdaten-Präfixe reichen nicht |
| multi-session | 0 % (0/3) | Aggregation über Sessions scheitert am Budget |
| single-session-preference | 0 % (0/4) | Implizite Präferenzen überleben die Kompression nicht |

## Diagnose

Das Muster (Einzelfakt gut, Aggregation/Implizites schlecht) passt zu einem
**zu knappen Retrieval-Budget**: top_k 3 × ≤2000 Token Injection kann für
Fragen, deren Evidenz über viele Sessions verteilt ist, nicht genug Original-
Material zurückholen; der ARCHIVE-Prompt allein verdichtet Präferenz-Nuancen
weg. Der Judge ist per Gegenprobe ausgeschlossen.

## Nächste Experimente

1. **Run 2 (läuft):** top_k 3→8, max_injection_tokens 2000→6000 — greift
   direkt die 0-%-Typen an; Kosten: geringere Ersparnis (erwartet ~5–6x).
2. Danach ggf.: kleinere Chunks (feineres Retrieval), Update-Semantik für
   knowledge-update (neuere Tripel bevorzugen), Frage-Datum stärker in den
   Retrieval-Query einbeziehen.

**Methodik-Hinweis für spätere Veröffentlichung:** lokaler LLM-Judge statt
GPT-4o (PO-Regel: keine Cloud-Keys) → Zahlen stets als „self-reported,
local judge" ausweisen. Stichprobengröße pro Typ nennen.
