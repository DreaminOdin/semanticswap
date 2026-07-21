# ADR-010: Active Memory Retrieval (LLM-gesteuerter Swap-In per Function Calling)

**Status:** Akzeptiert · **Datum:** 2026-07-15

## Kontext
Swap-In v1 (ADR-008) lädt Snippets automatisch per Vektorsuche auf die letzte
User-Frage. Das PAD (§5.1) empfiehlt zusätzlich den agentischen Weg: Das
Haupt-LLM erhält ein Tool und entscheidet selbst, wann es archivierte Details
braucht - z. B. wenn es im ARCHIVE-Prompt eine Segment-Referenz liest.
Risiko: Proxy-eigene Tools dürfen nicht mit Tools des Harness kollidieren
(offener Punkt #3 der PAD-Analyse).

## Entscheidung
1. Der Proxy injiziert das Tool **`retrieve_archived_memory`**
   (Parameter: `query` und/oder `segment_id`) in den Upstream-Request -
   aber nur wenn ALLE Bedingungen gelten:
   - die Session hat einen Archivstand,
   - der Request ist **nicht** gestreamt (v1),
   - der Client bringt **keine eigenen Tools** mit (Kollisionsvermeidung),
   - `retrieval.active_tool: true` (Config).
2. Ruft das Modell das Tool auf, löst der Proxy den Aufruf **intern** auf
   (Segment-Lookup bzw. Vektorsuche, Token-Budget aus ADR-008) und ruft das
   Modell mit dem Tool-Ergebnis erneut auf - maximal **2 Runden**, danach wird
   die letzte Antwort durchgereicht.
3. Der Tool-Zyklus ist für den Client **unsichtbar**: Tool-Messages werden
   nie in der Session-Historie persistiert und nie an den Client gesendet;
   der Client erhält nur die finale Assistant-Antwort.
4. Antworten ohne Tool-Aufruf werden unverändert durchgereicht.

## Konsequenzen
- Detailtreue steigt, Token-Verbrauch sinkt (nur angeforderte Segmente werden
  geladen), Latenz steigt bei Tool-Nutzung um eine Modell-Runde.
- Harnesses mit eigenen Tools nutzen weiterhin nur den passiven Swap-In -
  dokumentierte, sichere Einschränkung (Aufhebung wäre ein eigenes ADR:
  Tool-Merging mit Namespace).
- Streaming + Tool-Zyklus folgt später (erfordert Puffern des ersten Chunks).
