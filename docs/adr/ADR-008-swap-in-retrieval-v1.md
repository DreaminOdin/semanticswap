# ADR-008: Swap-In v1 - automatischer Vektor-Lookup mit Token-Budget

**Status:** Akzeptiert · **Datum:** 2026-07-15

## Kontext
Nach dem Swap-Out (M3) kennt das Haupt-LLM archivierte Inhalte nur noch als
komprimierte Summaries im ARCHIVE-Prompt. Fragt der User nach Details, müssen
die referenzierten Original-Transkripte zurückgeladen werden (PAD Phase 2).
Das PAD nennt zwei Wege: automatischer Lookup pro Request und agentisches
Active Retrieval per Function Calling (PAD §5.1).

## Entscheidung
Swap-In v1 ist der **automatische Weg** (Active Retrieval folgt in M6):

1. Pro Request einer Session mit Archivstand wird die **letzte User-Message**
   als Query embedded und gegen den Vektor-Store der Session gesucht
   (`top_k`, `min_score` konfigurierbar).
2. Treffer werden als **Original-Volltext-Segmente** in eine temporäre
   System-Message injiziert, direkt nach dem ARCHIVE-Prompt.
3. Ein **Token-Budget** (`max_injection_tokens`) begrenzt die Injection;
   das letzte Segment wird notfalls gekürzt, Segmente unter ~50 Token
   Restbudget entfallen.
4. Die Injection ist **flüchtig**: Sie wird nie in der Session-Historie
   persistiert und bei jedem Request neu berechnet.
5. Retrieval ist strikt **best effort**: Jeder Fehler (Embedding-Provider down,
   leerer Store) degradiert zu "keine Injection" - der Request läuft immer weiter.

## Konsequenzen
- Detailfragen zu archivierten Themen funktionieren ohne Client-Änderung.
- Kosten: ein Embedding-Call pro Request auf archivierten Sessions; akzeptiert,
  da Embeddings um Größenordnungen billiger sind als eingesparte Kontext-Token.
- Reine Vektorsuche findet nur semantisch Ähnliches (PAD §5.4);
  Graph-Expansion und LLM-gesteuertes Nachladen bleiben M6.
