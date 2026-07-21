"""Prompt-Templates: Worker-Prompts und das ARCHIVE-v2-Injektions-Template (PAD §5.5)."""
from __future__ import annotations

SUMMARIZATION_PROMPT = """You are a compression sub-agent in a memory proxy.
Summarize the following conversation segment into a highly condensed semantic core.
Keep: decisions, facts, goals, blockers, user preferences, technical details.
Drop: smalltalk, filler, repetition.

First line of your answer MUST be exactly `PRIORITY: high` or `PRIORITY: low`.
- high: facts about the user, project goals, decisions, reference data
- low: smalltalk, transient debugging steps, filler content
Then the summary, max 120 words. Nothing else.

SEGMENT:
{segment}
"""

ENTITY_EXTRACTION_PROMPT = """You are an entity-extraction sub-agent in a memory proxy.
Extract entities and their relations from the conversation segment below as
knowledge-graph triples. Respond with a JSON array only, no prose:
[{{"subject": "...", "predicate": "...", "object": "..."}}]
Max 15 triples. If nothing meaningful is found, respond with [].

SEGMENT:
{segment}
"""

PROFILE_PROMPT = """You are a profile-distillation sub-agent in a memory proxy.
From the conversation summaries below, distill a compact standing profile of the
USER: stable facts, preferences, recurring goals, constraints. Present tense,
terse bullet-style, max 100 words. Only durable user attributes — no one-off
events, no assistant details. If nothing durable, respond with an empty line.

SUMMARIES:
{summaries}
"""

DECOMPOSE_PROMPT = """The user's question may require combining facts from
several past moments. Break it into 1-{maxn} focused sub-questions, each
searchable on its own. If the question is already simple, return it unchanged
as a single line. One sub-question per line, numbered. No other text.

Question: {query}"""

RERANK_PROMPT = """You rank archived memory snippets by how well they help
answer the user's query. Consider each snippet's actual content, not just
keyword overlap.

Query: {query}

Snippets:
{snippets}

Respond with ONLY the snippet numbers from most to least relevant, comma-
separated (e.g. "3, 1, 2"). Include every number exactly once. No other text."""

ARCHIVE_HEADER = "### SYSTEM MEMORY COMPRESSION (ARCHIVE v2) ###"

_MAX_TRIPLE_LINES = 40
_MAX_SUMMARY_CHARS = 400

RETRIEVAL_HEADER = "### RETRIEVED ARCHIVE SNIPPETS (ON-DEMAND SWAP-IN) ###"


def build_retrieval_prompt(parts: list) -> str:
    """parts: Liste von (Segment, ggf. gekürzter Text, Score)."""
    lines = [
        RETRIEVAL_HEADER,
        "The user's latest message may refer to archived content. The memory "
        "proxy retrieved these original transcript excerpts:",
    ]
    for seg, text, score in parts:
        lines.append(f"--- #{seg.id} (messages {seg.start_idx}-{seg.end_idx}, "
                     f"relevance {score:.2f}) ---")
        lines.append(text)
    return "\n".join(lines)


def _apply_temporal_supersede(triples: list) -> list[tuple[str, str, str]]:
    """Iteration C: bei gleichem (Subjekt, Prädikat) gewinnt das Tripel mit der
    höchsten Recency (spätestes Segment). Erwartet 4er-Tupel (s,p,o,recency);
    3er-Tupel (ohne Recency) werden unverändert durchgereicht."""
    best: dict[tuple[str, str], tuple[int, tuple[str, str, str]]] = {}
    passthrough: list[tuple[str, str, str]] = []
    order: list[tuple[str, str]] = []
    for t in triples:
        if len(t) < 4:
            passthrough.append((t[0], t[1], t[2]))
            continue
        s, p, o, recency = t[0], t[1], t[2], t[3]
        key = (s.lower(), p.lower())
        if key not in best:
            order.append(key)
        if key not in best or recency >= best[key][0]:
            best[key] = (recency, (s, p, o))
    return [best[k][1] for k in order] + passthrough


def build_archive_prompt(segments: list, triples: list,
                         low_priority_visible: int = 3,
                         temporal_supersede: bool = False,
                         user_profile: str | None = None,
                         entity_resolution: bool = False) -> str:
    """Heuristischer Synthesizer-Schritt: konsolidiert Segment-Summaries und
    Graph-Tripel in das standardisierte Injektions-Template.

    Garbage Collector (ADR-011): Low-Priority-Segmente außerhalb der letzten
    `low_priority_visible` Segmente wandern ins Deep Archive (Sammelzeile,
    weiterhin per Retrieval abrufbar).

    entity_resolution (Prio 2): Entitäts-Varianten auf eine Oberflächenform
    kanonisieren, BEVOR dedupliziert/verdrängt wird.
    temporal_supersede (Iteration C): neuerer Fakt verdrängt älteren bei
    gleichem Subjekt+Prädikat; erwartet dann 4er-Tupel (s,p,o,recency)."""
    if entity_resolution:
        from .memory.entities import resolve_triples
        triples = resolve_triples(triples)
    if temporal_supersede:
        triples = _apply_temporal_supersede(triples)
    else:
        triples = [(t[0], t[1], t[2]) for t in triples]  # auf 3er normalisieren
    recent_ids = {seg.id for seg in segments[-max(0, low_priority_visible):]}
    visible = [seg for seg in segments
               if getattr(seg, "priority", "high") != "low" or seg.id in recent_ids]
    deep_archived = len(segments) - len(visible)

    lines = [
        ARCHIVE_HEADER,
        "The following is a structured state of the earlier conversation "
        "(compressed by the memory proxy):",
    ]
    if user_profile and user_profile.strip():
        # Immer prominent oben: das stehende Nutzerprofil (Iteration B).
        lines.append("- [User Profile] (durable facts & preferences):")
        lines.append(f"  {user_profile.strip()}")
    lines.append("- [Active Topics]:")
    for seg in visible:
        summary = (seg.summary or "").strip() or "(no summary available)"
        if len(summary) > _MAX_SUMMARY_CHARS:
            summary = summary[:_MAX_SUMMARY_CHARS] + "…"
        lines.append(f"  * Ref: #{seg.id} (messages {seg.start_idx}-{seg.end_idx}) "
                     f"-> Summarized: {summary}")
    if deep_archived:
        lines.append(f"  * ({deep_archived} low-priority segment(s) moved to deep "
                     f"archive - retrievable on demand)")
    if triples:
        lines.append("- [Known Entities & Relations]:")
        seen = set()
        emitted = 0
        for s, p, o in triples:
            key = (s.lower(), p.lower(), o.lower())
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"  * ({s}) -> {p} -> ({o})")
            emitted += 1
            if emitted >= _MAX_TRIPLE_LINES:
                lines.append(f"  * … ({len(triples) - emitted} weitere Relationen "
                             f"im Graph, per Retrieval abrufbar)")
                break
    lines.append(
        "- Full original transcripts of each #segment are archived and can be "
        "retrieved on demand."
    )
    return "\n".join(lines)
