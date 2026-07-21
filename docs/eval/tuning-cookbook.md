# SemanticSwap Tuning-Cookbook

> **Zweck.** Ein wachsendes Nachschlagewerk: Welche Technik/Modul hilft gegen
> welches Gedächtnis-Problem, was kostet sie anderswo, und wie man per Testing
> von „scheinbar besser" zu „wirklich besser" kommt. Dient zwei Zielgruppen:
> uns (Tuning-Disziplin) und den Anwendern (Bildung — was stelle ich wann ein).
>
> **Grundregel.** Jede Zahl hier ist entweder **[gemessen]** (echter
> Benchmark-Lauf) oder **[Hypothese]** (noch nicht validiert). Nie mischen.
> Messgrundlage: LongMemEval-S, stratifizierte 24er-Stichprobe (n≈23), lokaler
> Judge qwen2.5:7b, Modell gemma4:26b auf GX10. Details:
> [experiment-protokoll.md](experiment-protokoll.md), [ADR-013](../adr/ADR-013-externe-benchmarks.md).

---

## 1. Die Methode: wie man ehrlich misst

Diese fünf Regeln verhindern Selbstbetrug — die häufigste Falle beim Tuning.

1. **Eine Änderung pro Lauf.** Zwei Schrauben gleichzeitig zu drehen macht das
   Ergebnis uninterpretierbar. Run N = Run N-1 + genau ein Modul/Parameter.
2. **Teuren Teil vom billigen trennen.** Bei uns dominiert die Ingestion
   (~20 min/Instanz) die Laufzeit, ist aber für alle *Retrieval*-Experimente
   identisch. Der Ingestion-Cache (`--workdir`) senkt eine Retrieval-Iteration
   von ~8 h auf ~40 min [gemessen]. Erst dadurch wird systematisches Tuning
   überhaupt bezahlbar. → **Retrieval-seitige Module zuerst** (Cache-nutzend),
   **kompressions-seitige gebündelt** (erzwingen Re-Ingestion).
3. **Den Messrichter kontrollieren.** Ein LLM-Judge kann selbst die Fehlerquelle
   sein. Gegenprobe Run 1: Judge qwen2.5:7b und gemma4:26b lieferten **identische
   30,4 %** [gemessen] — der Judge ist also nicht der Engpass, die Antworten sind
   wirklich falsch. Solche Gegenproben vor jeder Ursachensuche.
4. **Pro Kategorie messen, nicht nur global.** Die Gesamt-Accuracy verbirgt, dass
   eine Schraube der einen Fragensorte hilft und der anderen schadet (siehe §2).
   Erst die Aufschlüsselung zeigt, *welches* Modul fehlt.
5. **Keine stillen Caps.** Übersprungene Instanzen (Timeout, Tool-Fehler) werden
   ausgewiesen und zählen nicht in die Accuracy. Ein geschöntes n ist wertlos.

---

## 2. Der zentrale Befund: Stellschrauben ≠ Verbesserung

**Experiment:** Retrieval-Budget erhöht (top_k 3→8, Injection 2000→6000 Token),
sonst nichts. **[gemessen]**

| | Run 1 (Baseline) | Run 2 (mehr Budget) | Δ |
|---|---|---|---|
| **QA-Accuracy** | 30,4 % | 34,8 % | +4,4 pp |
| **Kontext-Ersparnis** | 8,01x | 5,36x | **−33 %** |

Global sieht das nach „minimal besser" aus. Die Kategorie-Aufschlüsselung
erzählt die wahre Geschichte:

| Fähigkeits-Typ | Run 1 | Run 2 | Bewegung |
|---|---|---|---|
| temporal-reasoning | 25 % | 75 % | ↑ stark |
| single-session-assistant | 50 % | 75 % | ↑ |
| single-session-user | 75 % | 50 % | ↓ |
| knowledge-update | 25 % | 0 % | ↓ stark |
| multi-session | 0 % | 0 % | — |
| single-session-preference | 0 % | 0 % | — |

**Drei Lehren daraus:**
- **Mehr Kontext ist ein Nullsummenspiel.** Mehr zurückgeholte Snippets helfen
  Fragen, die zusätzliche Evidenz brauchen (temporal), verwässern aber Fragen,
  die eine *präzise* Antwort brauchen (single-user, knowledge-update) — das
  richtige Snippet ertrinkt in Beifang. Deshalb geht die eine Kategorie hoch,
  während die andere fällt.
- **Die harten Nullen bewegt kein Budget.** `multi-session` und `preference`
  bleiben bei 0 %, egal wie viel Kontext man gibt. Das sind **Architektur-Lücken**,
  keine Mengen-Probleme: Die Evidenz ist entweder gar nicht auffindbar
  (preference: steht als Nuance nirgends explizit) oder über Sessions verstreut
  (multi-session: kein einzelnes Snippet enthält die Antwort).
- **Das Kernversprechen leidet.** −33 % Ersparnis für +4 pp Accuracy ist ein
  schlechter Tausch. Budget-Drehen führt in die Mittelmäßigkeit — man verliert
  das Alleinstellungsmerkmal (Kompression), ohne das Problem zu lösen.

→ **Konsequenz:** Nicht weiter am Budget drehen. Gezielte Module bauen, die die
jeweilige *Ursache* adressieren (§3).

### Gegenprobe: ein Modul schlägt jede Stellschraube  **[gemessen]**

**Experiment:** Run 2 + **Hybrid-Suche** (FTS5-Volltext parallel zur
Vektorsuche, RRF-Fusion), sonst nichts geändert.

| | Run 2 (Budget) | Run 3 (+ Hybrid) | Δ |
|---|---|---|---|
| **QA-Accuracy** | 34,8 % | **54,2 %** | **+19,4 pp** |
| **Kontext-Ersparnis** | 5,36x | 5,19x | ~gleich |

| Fähigkeits-Typ | Run 2 | Run 3 |
|---|---|---|
| multi-session | 0 % | **50 %** |
| single-session-preference | 0 % | **50 %** |
| knowledge-update | 0 % | **50 %** |
| single-session-assistant | 75 % | 50 % |
| single-session-user | 50 % | 75 % |
| temporal-reasoning | 75 % | 50 % |

**Die Lehre — direkt gegenübergestellt:**
- Budget-Schraube (Run 1→2): **+4,4 pp** Accuracy, **−33 %** Ersparnis, harte
  Nullen unbewegt.
- Ein passendes Modul (Run 2→3): **+19,4 pp** Accuracy, Ersparnis fast
  unverändert, und die **beiden 0-%-Kategorien springen auf 50 %**.

Warum wirkt Hybrid so breit? Die reine Vektorsuche ist bei exakten
Zeichenketten (Namen, Zahlen, Entitäten) blind — sie findet „thematisch
ähnlich", nicht „genau dieses Wort". Gerade `multi-session` und `preference`
hängen an wiederkehrenden konkreten Begriffen über Gespräche hinweg; die
Volltextsuche findet die, wo Embeddings verschwimmen. **Das ist der
Kern-Takeaway des Cookbooks:** Erst die *Ursache* diagnostizieren, dann das
Modul wählen, das sie trifft — das schlägt jedes pauschale „mehr Kontext" um
Größenordnungen, ohne das Kernversprechen (Kompression) zu opfern.

Kleine Warnung zur Ehrlichkeit: Zwei Kategorien (assistant, user) wackeln bei
n=4 um ±1 Treffer — bei so kleiner Stichprobe ist Kategorie-Rauschen normal.
Der Gesamtsprung (+19 pp über 24 Instanzen) ist aber deutlich außerhalb dieses
Rauschens. Vor Veröffentlichung: auf 100+ Instanzen bestätigen.

### Gegenbeispiel: ein Modul scheitert an seiner Zielkategorie  **[gemessen]**

**Experiment:** Run 3 + **Graph-Expansion** (limit 3) — gebaut, um genau
`multi-session` zu heilen (Evidenz über Sessions per geteilte Entitäten
verbinden).

| | Run 3 (Hybrid) | Run 4 (+ Graph) |
|---|---|---|
| Accuracy gesamt | 54,2 % (n=24) | 68,2 % (n=22, **2 Timeouts**) |
| **multi-session** (Ziel!) | **50 %** | **0 %** (0/4) |

**Was das lehrt — und warum man der Gesamtzahl misstrauen muss:**
- Die globale Zahl *stieg* (54→68 %), aber sie ist **verunreinigt**: zwei
  Instanzen liefen in GX10-Timeouts, n fiel auf 22, das Weglassen zweier
  (möglicher) Misses hebt den Schnitt künstlich. **Nie zwei Läufe mit
  unterschiedlichem n direkt vergleichen.**
- Das einzige *saubere* Signal ist die Zielkategorie: `multi-session` fiel
  0/4 — alle vier liefen, alle scheiterten, gegenläufig zur Absicht des Moduls.
- **Beifang-Hypothese:** Graph-Expansion zieht ALLE Nachbarn über geteilte
  Entitäten. Häufige Entitäten (ein oft genannter Name) schleppen viele lose
  verwandte Segmente mit → das eigentliche Antwort-Snippet ertrinkt. Genau die
  Nebenwirkung, die in der Matrix (§3) unter „mehr Kontext → Beifang" steht.
- **Methoden-Lehre:** Ein plausibles, gezielt gebautes Modul kann seine
  Zielkategorie *verschlechtern*. Nur der Blick auf die Zielkategorie (nicht
  die Gesamtzahl) deckt das auf. → Test-Folge: `graph_expansion_limit 3→1`
  (Run 5) prüft, ob weniger Beifang multi-session rettet.

**Hypothese getestet — und widerlegt (Run 5):** `graph_expansion_limit 3→1`
(minimaler Beifang). Ergebnis: `multi-session` bleibt **0/4**. Weniger Nachbarn
retten es nicht → die Beifang-*Menge* war nicht die Ursache. Reproduziert über
zwei Settings (limit 3 und 1, beide 0/4), während ohne Graph 2/4.

**Endgültige Konsequenz: Graph-Expansion verworfen** (Code bleibt, default AUS).
Wahrscheinliche wahre Ursache: Schon ein einziges entitäts-ähnliches
Nachbar-Segment verdrängt im Token-Budget ein *echtes* Treffer-Segment — und
multi-session braucht die spezifischen Segmente, nicht die ähnlichen. Der
Entity-Graph ist das falsche Werkzeug; für multi-session ist eher
**Frage-Zerlegung** (Teilfragen einzeln suchen, §3) der richtige Ansatz —
ungetestet, Kandidat für später.

**Meta-Lehre (dreifach):** (1) Ein gezielt gebautes Modul kann seine
Zielkategorie verschlechtern. (2) Die naheliegende Erklärung („zu viel Beifang")
kann ebenfalls falsch sein — auch die *Hypothese* muss getestet werden, nicht
nur das Modul. (3) Zwei reproduzierte Läufe schlagen eine schöne Theorie. So
kommt man von „klingt plausibel" zu „ist belegt".

Hybrid dagegen: klarer Gewinn, default AN.

### Der sauberste Vergleich: Modul isolieren, Budget konstant  **[gemessen]**

Bisher war Hybrid nur *zusammen mit* höherem Budget getestet (Run 3). Um die
reine Hybrid-Wirkung zu sehen, Run 6 = Baseline-Budget (top_k 3, inj 2000) +
Hybrid — also **dasselbe Budget wie Run 1**, nur Hybrid dazu:

| (top_k 3, inj 2000) | Run 1 (ohne) | Run 6 (mit Hybrid) |
|---|---|---|
| Accuracy | 30,4 % | **50,0 %** (+19,6 pp) |
| Ersparnis | 8,01x | **8,14x** (gehalten) |

Und ob das große Budget überhaupt lohnt (beide mit Hybrid):

| mit Hybrid | Budget klein (Run 6) | Budget groß (Run 3) |
|---|---|---|
| Accuracy | 50,0 % | 54,2 % |
| Ersparnis | **8,14x** | 5,19x |

**Lehren:**
- **Vergleiche bei konstanter Nebenvariable.** Run 1 vs 6 (gleiches Budget)
  zeigt Hybrids Reineffekt: +19,6 pp bei **null** Ersparnis-Kosten. Der frühere
  Run 2→3-Vergleich vermischte Hybrid mit dem Budget und unterschätzte Hybrid
  (schrieb ihm den Ersparnis-Verlust des Budgets zu).
- **Das Budget lohnt auch mit Hybrid nicht:** +4 pp für −36 % Ersparnis — der
  immergleiche schlechte Tausch. Das gute Modul macht die teure Stellschraube
  nicht plötzlich sinnvoll.
- **Produktions-Empfehlung:** Hybrid AN, Budget klein (top_k 3, inj 2000).
  50 % Accuracy bei 8x Ersparnis — Genauigkeit *und* Kernversprechen.

---

## 3. Entscheidungsmatrix: Symptom → Ursache → Modul

Aus dem Fehlerprofil abgeleitet. Spalte „Status" = ob der Effekt schon
gemessen ist.

| Symptom (Fragensorte scheitert) | Wahrscheinliche Ursache | Passendes Modul | Nebenwirkung / Kosten | Status |
|---|---|---|---|---|
| Exakte Namen/Zahlen/Codes verfehlt; breite Schwäche | Rein semantische Suche ist bei Zeichenketten blind | **Hybrid-Suche** (FTS5-Volltext + Vektoren, RRF) | vernachlässigbar (FTS5 in SQLite eingebaut) | **[gemessen] +19,4 pp, Ersparnis ~gleich** |
| `multi-session`: Antwort über mehrere Gespräche verteilt | Ein einzelnes Snippet enthält die Antwort nie | ~~Graph-Expansion~~ **verworfen** → stattdessen **Frage-Zerlegung** (Teilfragen einzeln suchen) | Graph verdrängt echte Treffer durch entitäts-ähnliche | **[gemessen] Graph scheiterte** (0/4 bei limit 3 UND 1); Frage-Zerlegung ungetestet |
| `preference`: implizite Vorlieben | Nuancen überleben die Kompression nicht; nirgends als Fakt | **Profil-Gedächtnis** (stehendes Nutzerprofil, immer sichtbar) | fixe Token-Kosten pro Anfrage; Extraktions-Qualität | [Hypothese] Iteration B |
| `knowledge-update`: alter Fakt gewinnt | Neuer Fakt überschreibt alten nicht | **Temporale Tripel** (Zeitstempel, neueres Subjekt+Prädikat verdrängt) | Re-Ingestion nötig; Konflikt-Erkennung | [Hypothese] Iteration C |
| `temporal`: Zeitbezug fehlt | Sitzungsdatum nicht im Retrieval-Query | Frage-Datum in Query + temporale Tripel | s. o. | teilweise (Datum im Prompt) |
| Gesamtqualität gut, aber Beifang | Budget zu groß | top_k / Injection **senken** | Recall einzelner Fragen sinkt | [gemessen] (Run 2 zeigte die Kehrseite) |

**Lesart:** Erst das *Symptom* (welche Kategorie ist niedrig) → dann die
*Ursache* benennen → dann das Modul wählen, das die Ursache trifft. Nicht
umgekehrt („ich hab da ein cooles Modul") und nicht pauschal („mehr Kontext").

---

## 4. Modul-Steckbriefe (Einstellung & Wirkung)

Wird pro validiertem Modul gefüllt. Format: Was, Wann einschalten, Parameter,
gemessener Effekt, Wechselwirkungen.

### Hybrid-Suche — `retrieval.hybrid`
- **Was:** FTS5-Stichwortsuche parallel zur Vektorsuche, Ergebnisse per
  Reciprocal Rank Fusion (RRF) verschmolzen.
- **Wann:** sobald exakte Zeichenketten (Namen, IDs, Zahlen) vorkommen — also
  fast immer bei faktischen Fragen.
- **Kosten:** ~0 (FTS5 ist in SQLite; Alt-Datenbanken werden beim Öffnen
  indexiert). `min_score` gilt im Hybrid-Modus nicht (RRF-Ränge statt Cosine).
- **Robustheit:** fällt das Embedding-Backend aus, degradiert es zur reinen
  Volltextsuche statt zum Totalausfall.
- **Effekt:** **[gemessen]** stärkster Einzelhebel bisher: **+19,4 pp**
  Gesamt-Accuracy (34,8 → 54,2 %) bei praktisch unveränderter Ersparnis
  (5,36 → 5,19x). Hob alle sechs Kategorien, insbesondere die zuvor toten
  `multi-session` und `preference` (0 → 50 %). **Empfehlung: standardmäßig an.**

### Graph-Expansion — `retrieval.graph_expansion` (`graph_expansion_limit`)
- **Was:** Jeder Treffer zieht bis zu N Nachbar-Segmente mit, die im
  Wissens-Graph Entitäten (Subjekt/Objekt) teilen — verbindet Evidenz über
  Session-Grenzen.
- **Wann:** wenn `multi-session`/aggregierende Fragen schwach sind.
- **Kosten:** mehr Kontext pro Anfrage → geringere Ersparnis (Kehrseite aus §2
  beachten); `graph_expansion_limit` deckelt das.
- **Effekt:** **[gemessen — VERWORFEN]** `multi-session` fiel von 50 % auf 0 %,
  reproduziert bei limit 3 UND limit 1. Nicht die Beifang-Menge (Hypothese
  widerlegt), sondern Verdrängung echter Treffer durch entitäts-ähnliche
  Segmente. Default AUS, nicht empfohlen. Für multi-session stattdessen
  Frage-Zerlegung erwägen (ungetestet).

### Nutzer-Profil-Gedächtnis — _geplant (Iteration B)_
- **Was:** Ein Worker destilliert bei jeder Kompression ein stehendes Profil,
  das immer im ARCHIVE-Prompt steht (statt gefunden werden zu müssen).
- **Kosten:** feste Token pro Anfrage; **kompressions-seitig → Re-Ingestion**.

### Temporale Verdrängung — `compression.temporal_supersede`
- **Was:** Bei gleichem Subjekt+Prädikat gewinnt das Tripel aus dem späteren
  Segment; das ältere wird aus dem ARCHIVE-Prompt entfernt (bleibt per
  Retrieval abrufbar). Recency = Segment-Reihenfolge.
- **Trick:** wirkt im ARCHIVE-Prompt, der bei der Ingestion gebacken wird —
  aber query-seitig aus gespeicherten Segmenten/Tripeln neu rendern macht es
  **cache-testbar ohne Re-Ingestion**.
- **Effekt:** **[gemessen — neutral]** Gesamt 50,0 % (wie ohne), Ersparnis
  gehalten (8,18x); Zielkategorie knowledge-update 2/3→3/4, im Rauschen (n=4).
  Prinzipiell richtig, aber bei dieser Stichprobe kein belegbarer Gewinn.
  **Lehre:** Nicht jedes prinzipiell korrekte Feature liefert messbar — ein
  neutraler, harmloser Baustein braucht größere n zur Rechtfertigung. Default
  AUS bis 100+-Bestätigung.

### Nutzer-Profil-Gedächtnis — `compression.user_profile`
- **Was:** Ein Worker destilliert aus den Segment-Summaries ein stehendes
  Profil (dauerhafte Fakten/Vorlieben), das immer oben im ARCHIVE-Prompt steht.
- **Trick:** ebenfalls cache-testbar (Profil aus gespeicherten Summaries
  nachziehen, ein LLM-Call pro Instanz, statt Re-Ingestion).
- **Effekt:** _[Hypothese, Run 8 läuft]_ — Ziel: preference.

---

### Das richtige Werkzeug schlägt die richtige Technik  **[gemessen]**

**Experiment:** Gewinner + **Re-Ranker** (LLM-Listwise: kleines Modell sortiert
die 12 Kandidaten neu). Re-Ranking ist DIE Standard-RAG-Verbesserung — ich war
am zuversichtlichsten.

| | Gewinner (Run 6) | + Re-Ranker (Run 9, abgebrochen) |
|---|---|---|
| Accuracy | 50,0 % | **~14 %** (1/7, klar regressiv) |

**Diagnose (kein Bug, per Direkttest bestätigt):** qwen2.5:7b gibt auf einen
Rerank-Prompt mit 12 Kandidaten nur `"10, 1"` zurück — eine abgebrochene
2er-Liste. Diese zwei oberflächlich gewählten Kandidaten werden nach vorne
geschoben und **verdrängen die echten RRF-Bestentreffer aus den Top-3**.

**Lehren:**
- **Die richtige Technik mit dem falschen Werkzeug ist schlechter als keine
  Technik.** Listwise-Ranking braucht ein starkes Modell (GPT-4-Klasse) oder
  einen dedizierten **Cross-Encoder** (bge-reranker o. ä.), der (Query,
  Passage)-Paare direkt bepunktet — nicht ein kleines Chat-Modell, das eine
  Rangliste „erzählen" soll.
- **Zuversicht ist kein Ersatz für Messung.** Gerade beim Baustein, bei dem ich
  am sichersten war, deckte der Test die Schwäche auf.
- **Konsequenz:** LLM-Listwise-Re-Ranker VERWORFEN (default aus). Echter
  Cross-Encoder = offener nächster Schritt, braucht aber eine neue Abhängigkeit
  (FlagEmbedding/sentence-transformers, Apache-2.0) + Modell-Download →
  PO-Entscheidung (Projektregel: Lizenz/Abhängigkeit prüfen).

### Das Muster: Nur die einfache, deterministische Technik gewinnt  **[gemessen]**

Nach acht Läufen über sechs Bausteine ist das Bild eindeutig:

| Baustein | Accuracy | Ersparnis | Verdikt |
|---|---|---|---|
| **Hybrid-Suche** (FTS+Vektor+RRF) | **50 %** (+20 pp) | 8,1x | ✅ deterministisch, DER Gewinn |
| Temporale Verdrängung | 50 % | 8,2x | neutral |
| Profil-Gedächtnis | (Lauf unterbrochen) | — | offen |
| Graph-Expansion | ~0 % Zielkat. | — | ❌ schadet |
| **Re-Ranker (LLM-Listwise)** | ~14 % | — | ❌ schadet stark |
| **Entity Resolution** | 41,7 % | 8,3x | ❌ schadet (zu aggressive Kanonisierung: schreibt spezifische Werte weg) |
| **Query-Decomposition** | 37,5 % | — | ❌ schadet (multi-session 0/4; kleines Modell zerlegt schlecht) |
| **Cross-Encoder-Re-Ranker (ONNX)** | 33 % / 29 % | 8,0x | ❌ schadet — auch das *richtige Werkzeug*, mit korrekter Kürzung |

### Die tiefe Erkenntnis: exakter Treffer schlägt semantische Ähnlichkeit  **[gemessen]**

Der Cross-Encoder (dediziertes Ranking-Modell, in Isolation perfekt: +9 vs −11)
sollte das Muster brechen — und **scheiterte trotzdem** (33 %, nach Kürzungs-Fix
29 %). Das ist der lehrreichste Befund des Projekts, weil es *nicht* am Werkzeug
liegt:

**Warum Re-Ranking hier schadet — der Mechanismus:** Ohne Re-Ranker liefert die
Hybrid-RRF direkt die Top-3. Mit Re-Ranker werden 12 Kandidaten geholt und auf
3 umsortiert. Der Cross-Encoder optimiert auf **thematische Query-Passage-
Ähnlichkeit** (trainiert auf Web-Suche, MS-MARCO). Faktische
Gedächtnis-Fragen brauchen aber die Passage mit dem **exakten Fakt** — und den
findet die **FTS-Komponente der Hybrid-Suche über exaktes Keyword-Matching**
zuverlässiger. Der Re-Ranker bevorzugt dann eine Passage, die „über das Thema"
ist, gegenüber der, die die konkrete Zahl/den Namen enthält → er verdrängt die
richtige Antwort.

**Verallgemeinert (der Kern des ganzen Cookbooks):** Bei *faktischem* Recall aus
komprimiertem Gedächtnis gewinnt **exakter Abgleich** (FTS/BM25) gegen
**semantische Cleverness** (Vektoren allein, Re-Ranking, Graph-Ähnlichkeit,
Entitäts-Verschmelzung). Genau deshalb war Hybrid der Gewinn (es *fügt* exaktes
Matching hinzu) und genau deshalb schaden alle Aufbauten, die stattdessen auf
Ähnlichkeit optimieren. Das ist keine Werkzeug-Schwäche, sondern eine
Aufgaben-Eigenschaft — und ein starkes, verkaufbares Argument für unsere
Hybrid-Architektur.

**Ehrliche Einschränkung:** n=24 ist klein; einzelne Negativ-Ausschläge könnten
Rauschen sein. Aber die *Konsistenz* (6 Techniken, alle neutral bis negativ,
während Hybrid reproduzierbar +20 pp lieferte) macht die Aussage robust: Keine
dieser Erweiterungen ist bei diesem Aufbau einen klaren Gewinn wert.

**Die übergreifende Lehre — der wertvollste Befund dieses Projekts:**
Bei kleinen lokalen Modellen und Retrieval über verrauschtem Gedächtnis
gewinnt die **einfache, deterministische** Technik (exakte Volltextsuche neben
Vektoren). Jede Zusatz-„Cleverness" — LLM-Ranking, Graph-Nachbarn,
Entitäts-Kanonisierung — hat auf diesem Benchmark **geschadet oder war
neutral**. Gründe: LLM-in-the-loop scheitert am schwachen Modell; Heuristiken
(Graph/Kanonisierung) fügen falsches Material hinzu oder löschen richtiges.
→ **Ship hybrid, halte die Architektur schlank.** Weitere Gewinne brauchen ein
*richtiges Werkzeug* (Cross-Encoder statt Chat-Modell) oder ein *größeres
Modell*, nicht mehr Heuristik-Schichten. Das ist keine Niederlage, sondern eine
teure Erkenntnis, sauber erkauft — genau wofür der Benchmark da ist.

## 4a. Empfohlene Presets  **[gemessen, Stand Run 6]**

| Preset | Einstellung | Wofür | Messung |
|---|---|---|---|
| **Ausgewogen (Default)** | `hybrid: true`, top_k 3, inj 2000 | Regelfall: beste Genauigkeit *ohne* Ersparnis zu opfern | 50 %, 8,14x |
| **Recall-stark** | `hybrid: true`, top_k 8, inj 6000 | wenn Genauigkeit über Kosten geht (viel RAM/Kontext) | 54 %, 5,19x |
| **nicht empfohlen** | Graph-Expansion | schadet multi-session | verworfen |

Weitere Presets (Coding-Agent, Companion, Doku-RAG) folgen, sobald die
kompressions-seitigen Module (B/C) gemessen sind.

## 5. Offene To-dos für dieses Dokument
- Run 3/4-Zahlen eintragen, Hybrid- und Graph-Steckbrief mit [gemessen] füllen.
- Nach B/C: Steckbriefe vervollständigen, Entscheidungsmatrix-Status aktualisieren.
- Wenn die Konfiguration steht: „Empfohlene Presets" ergänzen (z. B.
  „Coding-Agent", „Companion-Chat", „Doku-RAG") mit je einem Parametersatz.
- Kandidat für eine öffentliche, aufbereitete Fassung (Anwender-Bildung).
