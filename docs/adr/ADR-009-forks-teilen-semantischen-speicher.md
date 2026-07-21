# ADR-009: Session-Forks teilen den semantischen Speicher (memory_id)

**Status:** Akzeptiert · **Datum:** 2026-07-15

## Kontext
Bei Historien-Edits/Retries forkt der Tracker die Session (ADR-003). Der Fork
erbte bisher den ARCHIVE-Prompt, aber Segmente, Graph-Tripel und Embeddings
blieben unter der ID der Ursprungs-Session gespeichert - Swap-In (ADR-008) und
Folge-Kompressionen liefen auf Forks ins Leere. Sichtbar wurde das beim Design
des M5-Evals: Jede unabhängige Recall-Frage auf demselben Verlauf erzeugt einen
Fork und hätte keinen Zugriff auf den Speicher gehabt.

## Entscheidung
Sessions erhalten eine **`memory_id`** (Default: die eigene Session-ID).
Forks erben die `memory_id` des Ursprungs. Alle Speicher-Artefakte (Segmente,
Tripel, Embeddings) werden unter der `memory_id` abgelegt und gelesen -
Kompression, Retrieval und Admin-API arbeiten auf dem geteilten Speicherraum.
Der Archiv-Zeiger `archived_upto` bleibt pro Session (Forks können
unterschiedlich weit archiviert sein).

## Konsequenzen
- Alle Zweige eines Verlaufs teilen sich additiv dasselbe Wissen; Retrieval
  funktioniert auf jedem Zweig.
- Segment-IDs sind über Message-Ranges deterministisch; überlappende
  Archivierung divergenter Zweige überschreibt idempotent (INSERT OR REPLACE).
- Bestehende Datenbanken werden per Mini-Migration (ALTER TABLE) ergänzt.
