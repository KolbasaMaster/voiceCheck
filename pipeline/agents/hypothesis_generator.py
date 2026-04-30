"""Генератор гипотез: на сэмпле корпуса предлагает N продажных подходов."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from ..llm_client import LLMClient
from ..preprocess import Dialogue


log = logging.getLogger(__name__)


@dataclass
class Hypothesis:
    id: str
    name: str
    methodology: str
    core_idea: str
    why_might_work: str
    key_moves: list[str]

    def to_block(self) -> str:
        moves = "\n".join(f"  • {m}" for m in self.key_moves) if self.key_moves else "  • —"
        return (
            f"Название: {self.name}\n"
            f"Методология: {self.methodology}\n"
            f"Суть: {self.core_idea}\n"
            f"Почему может сработать: {self.why_might_work}\n"
            f"Ключевые приёмы:\n{moves}"
        )

    @classmethod
    def from_dict(cls, d: dict, fallback_id: str) -> "Hypothesis":
        return cls(
            id=str(d.get("id") or fallback_id),
            name=str(d.get("name") or "—"),
            methodology=str(d.get("methodology") or "—"),
            core_idea=str(d.get("core_idea") or "—"),
            why_might_work=str(d.get("why_might_work") or "—"),
            key_moves=[str(x) for x in (d.get("key_moves") or [])],
        )


def _format_corpus(
    dialogues: list[Dialogue],
    max_dialogues: int,
    head_turns: int,
    tail_turns: int,
) -> str:
    chunks = []
    for d in dialogues[:max_dialogues]:
        body = d.head_tail_text(head_turns, tail_turns)
        chunks.append(f"[diag {d.id} | исход: {d.outcome}]\n{body}")
    return "\n\n---\n\n".join(chunks)


async def generate_hypotheses(
    client: LLMClient,
    sample: list[Dialogue],
    n_hypotheses: int,
    *,
    max_dialogues_in_prompt: int = 60,
    head_turns: int = 6,
    tail_turns: int = 4,
    max_length: int = 6000,
) -> list[Hypothesis]:
    corpus = _format_corpus(sample, max_dialogues_in_prompt, head_turns, tail_turns)
    system, user = client.render(
        "HypothesisGen",
        n_hypotheses=n_hypotheses,
        corpus=corpus,
    )
    data = await client.chat_json(
        system=system, user=user, max_length=max_length, temperature=0.8
    )
    if not data or "hypotheses" not in data:
        log.warning("HypothesisGen вернул не-JSON или без ключа `hypotheses`")
        return []
    items = data.get("hypotheses") or []
    return [Hypothesis.from_dict(x, fallback_id=f"h{i+1}") for i, x in enumerate(items[:n_hypotheses])]
