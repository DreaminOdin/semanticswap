# ADR-014: Auth v2 — Geräte-Cookies, Tailnet-Vertrauen, Zugriffs-Logbuch

**Status:** Akzeptiert · **Datum:** 2026-07-19

## Kontext
Der Proxy ist öffentlich erreichbar (apimanufaktur.de → Tailscale-Funnel →
GX10). Ein einzelnes Basic-Auth-Passwort war der einzige Schutz; Scanner
klopfen die Funnel-URL laufend ab. PO-Ziel (19.07.): Ivo per Gerätefaktor
ohne Passwort, Familie (Jan, später weitere) mit zweitem Faktor und nur
ausgewählten Bereichen, plus Nachvollziehbarkeit fremder Zugriffe.

## Entscheidung
1. **Geräte-Cookie** (`semanticswap_device`): HMAC-signiert (Server-Secret
   in `data/auth_secret.key`, Rotation = globaler Logout), Rolle + Ablauf
   (90 d) + Nonce im Payload. Login-Seite `/ui/login` stellt es aus.
2. **Rollen:** `family` → Chat + Live-Ansicht (`/ui/studio|flow|chat|events`,
   `/v1/chat/completions`, `/v1/models`); `admin` (Basic/Bearer/Tailnet) →
   alles inkl. `/ui`-Sessions, `/admin/*`, `/ui/access`.
3. **Tailnet-Vertrauen** (`trust_tailnet` + `tailnet_users`): Quelle im
   CGNAT-Bereich 100.64/10 UND `tailscale whois` liefert einen erlaubten
   Account → Admin ohne Passwort. Geteilte/fremde Nodes werden NICHT
   automatisch vertraut.
4. **Login-Bremse:** Fehlversuche pro IP, exponentielle Verzögerung.
5. **Zugriffs-Logbuch** (`data/access-log.jsonl`, Ringpuffer + Rotation):
   jeder Zugriff außerhalb des vertrauten Tailnets mit Herkunfts-IP, Weg,
   Pfad, Ergebnis. Admin-Ansicht `/ui/access` (HTML) + `/admin/access-log`
   (JSON). Abgewiesene Zugriffe zusätzlich live als `access_denied`-Event.

## Kritische Sicherheitsregel (verifiziert 19.07.)
Tailnet-Vertrauen basiert AUSSCHLIESSLICH auf der TCP-Quelle
(`request.client.host`), NIE auf `X-Forwarded-For`. Deshalb läuft uvicorn
mit `proxy_headers=False, forwarded_allow_ips=[]` — sonst überschreibt es
die Client-IP aus dem (vom Funnel als trusted forwarder akzeptierten)
Header, und ein gefälschtes `X-Forwarded-For: <Ivos-Tailnet-IP>` würde
Admin-Zugang verschaffen. Das Logbuch liest den Header selbst, aber nur zur
Anzeige. Live-Test: gefälschter Header → 401; echtes Tailnet-Gerät → 200.

## Konsequenzen
- Jede Prüfung läuft auf dem GX10 (die Funnel-URL umgeht die Vercel-Edge).
- Offene Kopplung: Solange die geteilte Vercel-Middleware unsere Pfade mit
  Basic-Auth abfragt, kommt der weitergereichte Header beim GX10 als Admin
  an — Jan wäre über die Domain Admin, nicht Family. Für echte Family-Rolle
  über die Domain müssten die SemanticSwap-Pfade an der Edge durchgereicht
  werden (eigenes Follow-up, geteilte Datei → Absprache mit CICERO).
- TOTP (RFC 6238) bleibt optional für später (Closed-Core-Kandidat).
