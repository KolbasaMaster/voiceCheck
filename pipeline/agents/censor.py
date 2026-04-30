"""Цензор: оценивает один симулированный диалог по нескольким методологиям."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ..llm_client import LLMClient
from .hypothesis_generator import Hypothesis
from .persona_extractor import Persona
from .simulator import SimResult


log = logging.getLogger(__name__)

METHODOLOGIES = ("SPIN", "Sandler", "Challenger", "BANT", "MEDDIC")


@dataclass
class CensorScore:
    methodology: str
    score: float
    comment: str


@dataclass
class CensorVerdict:
    hypothesis_id: str
    persona_id: str
    outcome: str  # sold | scheduled | refused | inconclusive
    scores: list[CensorScore]
    strengths: list[str]
    weaknesses: list[str]

    @property
    def avg_score(self) -> float:
        if not self.scores:
            return 0.0
        return round(sum(s.score for s in self.scores) / len(self.scores), 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "persona_id": self.persona_id,
            "outcome": self.outcome,
            "avg_score": self.avg_score,
            "scores": [
                {"methodology": s.methodology, "score": s.score, "comment": s.comment}
                for s in self.scores
            ],
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
        }


def _parse_scores(payload: dict[str, Any]) -> list[CensorScore]:
    raw = payload.get("scores") or {}
    out: list[CensorScore] = []
    for m in METHODOLOGIES:
        item = raw.get(m) or {}
        try:
            score = float(item.get("score") or 0)
        except (TypeError, ValueError):
            score = 0.0
        out.append(CensorScore(
            methodology=m,
            score=max(0.0, min(10.0, score)),
            comment=str(item.get("comment") or ""),
        ))
    return out


async def evaluate(
    client: LLMClient,
    hypothesis: Hypothesis,
    persona: Persona,
    sim: SimResult,
) -> CensorVerdict:
    system, user = client.render(
        "CensorEval",
        hypothesis_block=hypothesis.to_block(),
        persona_block=persona.to_block(),
        transcript=sim.transcript(),
    )
    data = await client.chat_json(
        system=system, user=user, max_length=2500, temperature=0.2
    )
    if not data:
        log.warning("Censor вернул не-JSON; ставим заглушку")
        return CensorVerdict(
            hypothesis_id=hypothesis.id,
            persona_id=persona.id,
            outcome="inconclusive",
            scores=[CensorScore(m, 0.0, "—") for m in METHODOLOGIES],
            strengths=[],
            weaknesses=["цензор не смог распарсить транскрипт"],
        )
    return CensorVerdict(
        hypothesis_id=hypothesis.id,
        persona_id=persona.id,
        outcome=str(data.get("outcome") or "inconclusive"),
        scores=_parse_scores(data),
        strengths=[str(x) for x in (data.get("strengths") or [])],
        weaknesses=[str(x) for x in (data.get("weaknesses") or [])],
    )
