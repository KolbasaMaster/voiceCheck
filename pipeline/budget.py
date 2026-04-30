"""Грубая оценка токен-бюджета прогона до запуска.

Позволяет до старта прикинуть, влезаем ли в пользовательский потолок токенов,
и по-человечески объяснить раскладку. Все цифры приблизительные — реальные
расходы зависят от длины ответов модели и квадратичного роста контекста в
симуляции.
"""
from __future__ import annotations

from dataclasses import dataclass


# Эмпирический коэффициент: ~1 токен на 2 символа кириллицы (BPE GigaChat).
CHARS_PER_TOKEN = 2.0


@dataclass
class BudgetEstimate:
    preprocess: int
    hypothesis_gen: int
    simulation: int
    censor: int
    reporter: int
    total: int

    def as_table(self) -> str:
        lines = [
            "Этап              Токены (≈)",
            "─" * 30,
            f"Препроцессинг     {self.preprocess:>10,}".replace(",", " "),
            f"Гипотезы          {self.hypothesis_gen:>10,}".replace(",", " "),
            f"Симуляция         {self.simulation:>10,}".replace(",", " "),
            f"Цензор            {self.censor:>10,}".replace(",", " "),
            f"Репортёр          {self.reporter:>10,}".replace(",", " "),
            "─" * 30,
            f"Итого             {self.total:>10,}".replace(",", " "),
        ]
        return "\n".join(lines)


def estimate(
    *,
    n_dialogues_sample: int,
    avg_turn_chars: float,
    n_hypotheses: int,
    n_personas: int,
    n_turns: int,
    methodologies: int = 5,
) -> BudgetEstimate:
    """Оценка по числу прогонов и квадратичному росту контекста симуляции."""
    turn_tokens = max(20, int(avg_turn_chars / CHARS_PER_TOKEN))

    preprocess = n_dialogues_sample * turn_tokens * 6  # на оба прохода (разметка/выборка)

    # Hypothesis Generator получает корпус-сэмпл целиком + промпт.
    hyp_in = n_dialogues_sample * turn_tokens * 8 + 2_000
    hyp_out = n_hypotheses * 400
    hypothesis_gen = hyp_in + hyp_out

    # Симуляция: для каждой реплики оба агента читают всю историю.
    # На каждом ходу контекст ≈ ход_токенов × (текущая_длина_истории).
    runs = n_hypotheses * n_personas
    history_grow = sum(range(1, n_turns + 1))  # 1+2+...+N
    sim_per_run = history_grow * turn_tokens * 2  # два агента
    simulation = runs * sim_per_run

    # Цензор оценивает каждый прогон по N методологиям.
    censor_per_run = (n_turns * turn_tokens) + methodologies * 300
    censor = runs * censor_per_run

    # Репортёр агрегирует: на каждую гипотезу читает все её прогоны (короткие
    # выжимки от цензора) + пишет саммари.
    reporter_in = n_hypotheses * (n_personas * 800 + 1_500)
    reporter_out = n_hypotheses * 1_200
    reporter = reporter_in + reporter_out

    total = preprocess + hypothesis_gen + simulation + censor + reporter
    return BudgetEstimate(
        preprocess=preprocess,
        hypothesis_gen=hypothesis_gen,
        simulation=simulation,
        censor=censor,
        reporter=reporter,
        total=total,
    )
