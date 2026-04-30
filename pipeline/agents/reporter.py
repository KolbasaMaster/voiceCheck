"""Репортёр: агрегирует все прогоны одной гипотезы в подробное саммари."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from ..llm_client import LLMClient
from .censor import CensorVerdict
from .hypothesis_generator import Hypothesis


log = logging.getLogger(__name__)


@dataclass
class HypothesisReport:
    hypothesis_id: str
    verdict: str  # promising | mixed | weak
    overall_score: float
    by_methodology: dict[str, dict[str, Any]]
    best_persona: str
    worst_persona: str
    what_worked: list[str]
    what_failed: list[str]
    recommendations: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "verdict": self.verdict,
            "overall_score": self.overall_score,
            "by_methodology": self.by_methodology,
            "best_persona": self.best_persona,
            "worst_persona": self.worst_persona,
            "what_worked": self.what_worked,
            "what_failed": self.what_failed,
            "recommendations": self.recommendations,
        }


async def report_hypothesis(
    client: LLMClient,
    hypothesis: Hypothesis,
    verdicts: list[CensorVerdict],
) -> HypothesisReport:
    if not verdicts:
        return HypothesisReport(
            hypothesis_id=hypothesis.id,
            verdict="weak",
            overall_score=0.0,
            by_methodology={},
            best_persona="—",
            worst_persona="—",
            what_worked=[],
            what_failed=["нет прогонов"],
            recommendations=[],
        )
    runs_payload = json.dumps([v.to_dict() for v in verdicts], ensure_ascii=False)
    system, user = client.render(
        "ReporterSummary",
        hypothesis_block=hypothesis.to_block(),
        runs_json=runs_payload,
    )
    data = await client.chat_json(
        system=system, user=user, max_length=3000, temperature=0.3
    )
    if not data:
        # Локальный фолбэк: считаем агрегаты без LLM, чтобы не терять прогон.
        return _local_fallback(hypothesis, verdicts)

    try:
        overall = float(data.get("overall_score") or 0)
    except (TypeError, ValueError):
        overall = 0.0
    return HypothesisReport(
        hypothesis_id=str(data.get("hypothesis_id") or hypothesis.id),
        verdict=str(data.get("verdict") or "mixed"),
        overall_score=round(overall, 2),
        by_methodology=dict(data.get("by_methodology") or {}),
        best_persona=str(data.get("best_persona") or "—"),
        worst_persona=str(data.get("worst_persona") or "—"),
        what_worked=[str(x) for x in (data.get("what_worked") or [])],
        what_failed=[str(x) for x in (data.get("what_failed") or [])],
        recommendations=[str(x) for x in (data.get("recommendations") or [])],
    )


def _local_fallback(
    hypothesis: Hypothesis, verdicts: list[CensorVerdict]
) -> HypothesisReport:
    avg = sum(v.avg_score for v in verdicts) / len(verdicts)
    best = max(verdicts, key=lambda v: v.avg_score)
    worst = min(verdicts, key=lambda v: v.avg_score)
    by_meth: dict[str, dict[str, Any]] = {}
    methodologies = [s.methodology for s in verdicts[0].scores]
    for m in methodologies:
        scores = [s.score for v in verdicts for s in v.scores if s.methodology == m]
        by_meth[m] = {"avg": round(sum(scores) / len(scores), 2), "note": "локальный fallback"}
    verdict = "promising" if avg >= 7 else "weak" if avg < 4 else "mixed"
    return HypothesisReport(
        hypothesis_id=hypothesis.id,
        verdict=verdict,
        overall_score=round(avg, 2),
        by_methodology=by_meth,
        best_persona=best.persona_id,
        worst_persona=worst.persona_id,
        what_worked=[],
        what_failed=["LLM не вернул валидный JSON для саммари"],
        recommendations=[],
    )
