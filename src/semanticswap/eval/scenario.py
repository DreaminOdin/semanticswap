"""Synthetisches Eval-Szenario (M5): lange Konversation mit verankerten Fakten.

Jeder Turn behandelt ein Thema und enthält genau einen prüfbaren Fakt
(Referenzcode). Die Recall-Fragen prüfen später, ob das Speichersystem
(Archiv, Graph, Swap-In) diese Fakten über die Kompression hinweg erhält.
"""
from __future__ import annotations

from dataclasses import dataclass

TOPICS = [
    "Datenbank-Migration",
    "Frontend-Redesign",
    "Deployment-Pipeline",
    "Auth-Konzept",
    "Monitoring-Setup",
    "Kostenplanung",
    "Datenschutz-Audit",
    "Team-Onboarding",
]


@dataclass
class Question:
    text: str
    expected: str
    topic: str


def build_scenario(n_topics: int = 6,
                   filler_sentences: int = 6) -> tuple[list[str], list[Question]]:
    n_topics = min(n_topics, len(TOPICS))
    turns: list[str] = []
    questions: list[Question] = []
    for i, topic in enumerate(TOPICS[:n_topics]):
        code = f"ZX-{i + 1}{i + 4}7"
        filler = " ".join(
            f"Detail {j + 1} zu {topic}: Aspekt {j + 1} wurde geprüft, bewertet "
            f"und dokumentiert."
            for j in range(filler_sentences)
        )
        turns.append(
            f"Lass uns über {topic} sprechen. {filler} "
            f"Wichtig: Der Referenzcode für {topic} lautet {code}."
        )
        questions.append(Question(
            text=f"Wie lautet der Referenzcode für {topic}?",
            expected=code,
            topic=topic,
        ))
    return turns, questions
