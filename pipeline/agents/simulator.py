"""Симуляция диалога: продавец (по гипотезе) ↔ клиент (по персоне).

Контекст-окно: каждому ходу скармливаем последние WINDOW реплик, а не всю
историю — иначе токен-бюджет растёт квадратично от длины диалога.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from ..llm_client import LLMClient
from .hypothesis_generator import Hypothesis
from .persona_extractor import Persona


log = logging.getLogger(__name__)

WINDOW = 12  # последние N реплик в контексте каждого хода

# Маркеры завершения диалога (если встретились в реплике продавца).
_FAREWELL = ("до свидания", "всего доброго", "хорошего дня", "удачи")
# Маркеры явного отказа клиента.
_REFUSE = ("не интересно", "нет, спасибо", "не нужно", "не подходит", "не звоните")


@dataclass
class SimTurn:
    speaker: str  # "Сотрудник" | "Клиент"
    text: str


@dataclass
class SimResult:
    hypothesis_id: str
    persona_id: str
    turns: list[SimTurn]
    stop_reason: str  # "farewell" | "refused_thrice" | "max_turns" | "error"

    def transcript(self) -> str:
        return "\n".join(f"{t.speaker}: {t.text}" for t in self.turns)


def _history_block(turns: list[SimTurn], window: int = WINDOW) -> str:
    if not turns:
        return "(диалог только начинается)"
    return "\n".join(f"{t.speaker}: {t.text}" for t in turns[-window:])


def _is_farewell(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _FAREWELL)


def _is_refusal(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _REFUSE)


async def simulate(
    client: LLMClient,
    hypothesis: Hypothesis,
    persona: Persona,
    *,
    max_turns: int,
    opener: str = "Алло.",
) -> SimResult:
    """Гонит один диалог. Открывает клиент репликой `opener` (как в реальном корпусе)."""
    turns: list[SimTurn] = [SimTurn(speaker="Клиент", text=opener)]
    refuse_streak = 0
    stop_reason = "max_turns"

    sales_sys, _ = client.render(
        "SalesAgentTurn",
        methodology=hypothesis.methodology,
        hypothesis_block=hypothesis.to_block(),
        history="(вставится при вызове)",
    )
    client_sys, _ = client.render(
        "ClientAgentTurn",
        persona_block=persona.to_block(),
        history="(вставится при вызове)",
    )

    for _ in range(max_turns - 1):
        # Сотрудник
        try:
            sales_text = await client.chat(
                system=sales_sys,
                user=f"История диалога на текущий момент:\n\n{_history_block(turns)}\n\nТвой следующий ход.",
                max_length=400,
                temperature=0.7,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("sales agent failed: %s", e)
            stop_reason = "error"
            break
        sales_text = (sales_text or "").strip()
        if not sales_text:
            stop_reason = "error"
            break
        turns.append(SimTurn(speaker="Сотрудник", text=sales_text))
        if _is_farewell(sales_text) and refuse_streak >= 2:
            stop_reason = "farewell"
            break
        if len(turns) >= max_turns:
            break

        # Клиент
        try:
            client_text = await client.chat(
                system=client_sys,
                user=f"История диалога на текущий момент:\n\n{_history_block(turns)}\n\nТвой следующий ход.",
                max_length=400,
                temperature=0.85,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("client agent failed: %s", e)
            stop_reason = "error"
            break
        client_text = (client_text or "").strip()
        if not client_text:
            stop_reason = "error"
            break
        turns.append(SimTurn(speaker="Клиент", text=client_text))
        if _is_refusal(client_text):
            refuse_streak += 1
            if refuse_streak >= 3:
                stop_reason = "refused_thrice"
                break
        else:
            refuse_streak = 0
        if len(turns) >= max_turns:
            break

    return SimResult(
        hypothesis_id=hypothesis.id,
        persona_id=persona.id,
        turns=turns,
        stop_reason=stop_reason,
    )
