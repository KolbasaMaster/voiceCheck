"""Извлечение N клиентских персон из корпуса."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from ..llm_client import LLMClient
from ..preprocess import Dialogue


log = logging.getLogger(__name__)


@dataclass
class Persona:
    id: str
    name: str
    business_type: str
    tone: str
    main_objections: list[str]
    buying_readiness: str
    speech_quirks: str

    def to_block(self) -> str:
        objs = ", ".join(self.main_objections) if self.main_objections else "—"
        return (
            f"Имя: {self.name}\n"
            f"Бизнес: {self.business_type}\n"
            f"Тон: {self.tone}\n"
            f"Готовность к покупке: {self.buying_readiness}\n"
            f"Типичные возражения: {objs}\n"
            f"Речевые особенности: {self.speech_quirks}"
        )

    @classmethod
    def from_dict(cls, d: dict, fallback_id: str) -> "Persona":
        return cls(
            id=str(d.get("id") or fallback_id),
            name=str(d.get("name") or "Клиент"),
            business_type=str(d.get("business_type") or "—"),
            tone=str(d.get("tone") or "нейтральный"),
            main_objections=[str(x) for x in (d.get("main_objections") or [])],
            buying_readiness=str(d.get("buying_readiness") or "medium"),
            speech_quirks=str(d.get("speech_quirks") or "—"),
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


async def extract_personas(
    client: LLMClient,
    sample: list[Dialogue],
    n_personas: int,
    *,
    max_dialogues_in_prompt: int = 60,
    head_turns: int = 6,
    tail_turns: int = 4,
    max_length: int = 6000,
) -> list[Persona]:
    corpus = _format_corpus(sample, max_dialogues_in_prompt, head_turns, tail_turns)
    system, user = client.render(
        "PersonaExtraction",
        n_personas=n_personas,
        corpus=corpus,
    )
    data = await client.chat_json(
        system=system, user=user, max_length=max_length, temperature=0.7
    )
    if not data or "personas" not in data:
        log.warning("PersonaExtraction вернул не-JSON или без ключа `personas`")
        return []
    items = data.get("personas") or []
    return [Persona.from_dict(x, fallback_id=f"p{i+1}") for i, x in enumerate(items[:n_personas])]
