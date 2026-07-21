# ADR-003: Session-Erkennung per Prefix-Hash-Kette, Pruning ist virtuell

**Status:** Akzeptiert · **Datum:** 2026-07-15

## Kontext
Die OpenAI-Chat-API ist stateless: Der Client schickt bei jedem Request den
vollständigen Verlauf. Der Proxy muss erkennen, welcher Teil des eingehenden
Verlaufs bereits komprimiert/archiviert wurde — ohne Client-Anpassung
(Harness-Agnostizismus). Dies ist die kritischste Lücke des ursprünglichen PAD.

## Entscheidung
1. **Virtuelles Pruning:** Der Client-Verlauf wird nie verändert. Der Proxy hält
   intern pro Session einen Zeiger `archived_upto`; beim Forwarding ersetzt er
   die archivierten Messages durch den kompakten ARCHIVE-Memory-Prompt und hängt
   die restlichen Messages im Original an.
2. **Wiedererkennung per Hash-Kette (Merkle-artig):** Pro Message
   `msg_hash = sha256(role + '\x00' + normalisierter Content)`, verkettet zu
   `h_i = sha256(h_{i-1} + msg_hash_i)`. Alle Kettenglieder werden mit
   `(chain_hash, session_id, msg_index)` persistiert. Ein eingehender Request
   wird über den **längsten bekannten Prefix** einer Session zugeordnet; bei
   Mehrdeutigkeit gewinnt die zuletzt aktive Session.
3. **Explizite Session-ID hat Vorrang:** Header `x-session-id` (oder das
   OpenAI-Feld `user`) überschreibt das Fingerprinting, wenn vorhanden.
4. **Fork bei Historien-Edit:** Divergiert der eingehende Verlauf vor
   `archived_upto`, wird eine neue Session (Fork) angelegt; der Archivstand der
   alten Session bleibt unangetastet.
5. **Nebenläufigkeit:** Pro Session ein `asyncio.Lock` für Statuswechsel;
   Kompression committet transaktional, Requests nutzen den letzten
   committeten Stand.

## Konsequenzen
- Proxy ist jederzeit abschaltbar ohne Datenverlust (Client hält den Vollverlauf).
- Clients, die selbst truncaten/summarizen, brechen den Prefix-Match → werden
  als neue Session behandelt und geloggt (dokumentierte Grenze).
- Content-Normalisierung (Whitespace, kanonische Serialisierung von
  Content-Part-Arrays) ist Pflicht vor dem Hashen.
