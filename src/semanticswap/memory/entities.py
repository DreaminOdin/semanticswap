"""Entity Resolution / Kanonisierung (Prio 2).

Deterministische Normalisierung von Entitäts-Oberflächenformen auf einen
kanonischen Schlüssel — repariert Graph-Dedup und temporale Verdrängung an der
Wurzel (unaufgelöste Entitäten waren die Ursache des Graph-Expansion-Fehlers).

Bewusst regelbasiert (kein LLM): Case-Faltung, Whitespace, Artikel vorne. Volle
Koreferenz ("sie" → "Anna") wäre ein LLM-Schritt bei der Extraktion — späterer
Ausbau; diese Stufe deckt die häufigsten Varianten cache-testbar ab.
"""
from __future__ import annotations

import re
from collections import Counter

# Artikel/Determinanten am Anfang (DE + EN), die keine Identität tragen.
_LEADING = re.compile(
    r"^(der|die|das|den|dem|des|ein|eine|einen|einem|einer|"
    r"the|a|an)\s+", re.IGNORECASE)
_WS = re.compile(r"\s+")


def canonical_key(entity: str) -> str:
    """Vergleichsschlüssel: klein, ohne führende Artikel, Whitespace normalisiert,
    Satzzeichen an den Rändern entfernt."""
    text = _WS.sub(" ", entity.strip()).strip(".,;:!?\"'()")
    prev = None
    while prev != text:  # ggf. mehrere Artikel ("die eine Anna" -> "Anna")
        prev = text
        text = _LEADING.sub("", text).strip()
    return text.casefold()


def resolve_triples(triples: list) -> list:
    """Mappt Subjekt/Objekt aller Tripel auf eine gemeinsame Oberflächenform je
    kanonischem Schlüssel (die häufigste, sonst die erste). Erhält die Recency
    (4er-Tupel) bzw. gibt 3er-Tupel zurück. Prädikate werden ebenfalls
    kanonisiert (nur für den Vergleich in nachgelagerten Schritten relevant)."""
    surfaces: dict[str, Counter] = {}
    for t in triples:
        for val in (t[0], t[2]):
            surfaces.setdefault(canonical_key(val), Counter())[val] += 1

    def surface(val: str) -> str:
        key = canonical_key(val)
        counter = surfaces.get(key)
        if not counter:
            return val
        # häufigste Oberfläche; bei Gleichstand die alphabetisch erste (stabil)
        return sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

    out = []
    for t in triples:
        s, p, o = surface(t[0]), t[1], surface(t[2])
        out.append((s, p, o, t[3]) if len(t) >= 4 else (s, p, o))
    return out
