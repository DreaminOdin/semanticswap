"""Iteration B (M8): Nutzer-Profil-Gedächtnis.

Diagnose LongMemEval: preference-Fragen (0 %) scheitern, weil implizite
Vorlieben als Nuancen über viele Gespräche verstreut sind und die Kompression
sie wegschleift. Fix: ein stehendes Profil, das bei der Kompression destilliert
und IMMER im ARCHIVE-Prompt sichtbar ist (statt gefunden werden zu müssen).
"""
import pytest

from semanticswap.compression.workers import ExtractionWorkers
from semanticswap.eval.fakes import OfflineLLM
from semanticswap.memory.store import Store
from semanticswap.prompts import build_archive_prompt


def test_store_profile_roundtrip_per_memory():
    store = Store(":memory:")
    store.set_profile("mem1", "Nutzer: Vegetarier, läuft 5 km.")
    store.set_profile("mem2", "Anderer Speicher.")
    assert store.get_profile("mem1") == "Nutzer: Vegetarier, läuft 5 km."
    assert store.get_profile("mem2") == "Anderer Speicher."
    assert store.get_profile("unbekannt") is None
    # Idempotent überschreibbar
    store.set_profile("mem1", "Aktualisiert.")
    assert store.get_profile("mem1") == "Aktualisiert."


def test_archive_prompt_includes_profile_prominently():
    prompt = build_archive_prompt([], [], user_profile="Nutzer: mag Tee, "
                                  "vegetarisch, Bestzeit 25:50.")
    assert "Nutzer: mag Tee" in prompt
    # Profil steht weit oben (vor den Active Topics), nicht am Ende vergraben
    assert prompt.index("mag Tee") < prompt.index("Active Topics")


def test_archive_prompt_without_profile_unchanged():
    prompt = build_archive_prompt([], [])
    assert "User Profile" not in prompt and "Nutzerprofil" not in prompt


@pytest.mark.asyncio
async def test_worker_builds_profile_from_summaries(test_config):
    # Der Profil-Worker destilliert aus den Segment-Summaries ein Profil.
    workers = ExtractionWorkers(OfflineLLM(), test_config.sub_agents)
    summaries = [
        "Der Referenzcode für Projekt Adler lautet ADL-042.",
        "Nutzer bevorzugt Tee gegenüber Kaffee.",
    ]
    profile = await workers.build_profile(summaries)
    assert isinstance(profile, str) and profile  # OfflineLLM liefert deterministisch
