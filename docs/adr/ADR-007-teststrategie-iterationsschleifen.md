# ADR-007: Teststrategie und Iterationsschleifen

**Status:** Akzeptiert · **Datum:** 2026-07-15

## Kontext
Das System hat nicht-deterministische Abhängigkeiten (LLMs). Qualität muss
trotzdem pro Meilenstein mess- und wiederholbar geprüft werden.

## Entscheidung
1. **Test-First pro Meilenstein:** Jeder Meilenstein hat eine Definition of Done
   in Form von Tests, die vor/mit der Implementierung geschrieben werden.
2. **Drei Testebenen:**
   - *Unit* (`tests/unit/`): deterministisch, LLM-Aufrufe durch `FakeLLM`
     gemockt, kein Netzwerk.
   - *Integration* (`tests/integration/`): Gateway-End-to-End über ASGI-Client
     mit Fake-Provider — voller Request→Session→Kompression→Pruning-Zyklus.
   - *Eval* (`tests/eval/`, ab M5): Recall-Benchmark gegen echte Modelle
     (lange Konversationen, Detailfragen), Metriken: Kompressions-Ratio,
     Recall-Quote, TTFT-Overhead.
3. **Iterationsschleife:** `pytest` läuft nach jeder Änderung (rot → grün →
   refactor). Ein Meilenstein gilt erst als abgeschlossen, wenn alle Tests der
   unteren Ebenen grün sind.
4. **Kein Merge über rote Tests**; Regressionen erzeugen zuerst einen
   reproduzierenden Test.

## Konsequenzen
- LLM-Nichtdeterminismus ist aus Unit/Integration vollständig herausgehalten;
  Qualität der Prompts wird separat im Eval gemessen.
- Leicht höherer Initialaufwand, dafür belastbare Iteration.
