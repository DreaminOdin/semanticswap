"""Prio 2 (Architektur-Ausbau): Entity Resolution / Kanonisierung.

Zep/Graphiti lösen Entitäten auf ("Anna", "Anna Schmidt", "die Anna" → ein
Knoten). Wir extrahieren rohe Tripel ohne Auflösung — DAS war die Ursache,
warum Graph-Expansion scheiterte. Deterministische Kanonisierung (Case,
Artikel, Whitespace) verbessert Dedup und temporale Verdrängung an der Wurzel.
"""
from semanticswap.memory.entities import canonical_key, resolve_triples
from semanticswap.prompts import build_archive_prompt


def test_canonical_key_normalizes_surface_forms():
    assert canonical_key("Anna") == canonical_key("anna")
    assert canonical_key("die Anna") == canonical_key("Anna")
    assert canonical_key("  Anna  ") == canonical_key("Anna")
    assert canonical_key("the Project") == canonical_key("Project")
    assert canonical_key("Anna") != canonical_key("Bert")


def test_resolve_triples_merges_and_picks_frequent_surface():
    triples = [
        ("die Anna", "wohnt_in", "Berlin", 1),
        ("Anna", "wohnt_in", "München", 5),
        ("anna", "mag", "Tee", 3),
    ]
    resolved = resolve_triples(triples)
    subjects = {t[0] for t in resolved}
    # Alle drei "Anna"-Varianten auf EINE kanonische Oberfläche gemappt
    assert len(subjects) == 1
    # temporale Verdrängung greift jetzt über die Varianten hinweg
    from semanticswap.prompts import _apply_temporal_supersede
    superseded = _apply_temporal_supersede(resolved)
    objects = {o for _, p, o in superseded if p == "wohnt_in"}
    assert objects == {"München"}   # Berlin verdrängt, obwohl "die Anna" vs "Anna"


def test_archive_prompt_entity_resolution_dedupes_variants():
    triples = [("die Anna", "mag", "Tee", 1), ("Anna", "mag", "Tee", 2)]
    prompt = build_archive_prompt([], triples, temporal_supersede=True,
                                  entity_resolution=True)
    # nur EINE Relation trotz zweier Oberflächenformen
    assert prompt.count("-> mag ->") == 1


def test_entity_resolution_off_keeps_variants_separate():
    triples = [("die Anna", "mag", "Tee", 1), ("Anna", "mag", "Tee", 2)]
    prompt = build_archive_prompt([], triples, temporal_supersede=False,
                                  entity_resolution=False)
    assert "die Anna" in prompt and prompt.count("-> mag ->") == 2
