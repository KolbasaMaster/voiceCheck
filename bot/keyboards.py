"""Инлайн-клавиатуры мастера настройки."""
from __future__ import annotations

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from .config import DEFAULTS

# Префиксы callback_data: <param>:<value> (value=число или 'last'/'custom').
CB_HYP = "hyp"
CB_PER = "per"
CB_TURN = "turn"
CB_BUDGET = "budget"
CB_LAST = "last"
CB_CUSTOM = "custom"


def _row(prefix: str, values: list[int]) -> list[InlineKeyboardButton]:
    return [InlineKeyboardButton(str(v), callback_data=f"{prefix}:{v}") for v in values]


def _ending(prefix: str, has_last: bool) -> list[InlineKeyboardButton]:
    btns: list[InlineKeyboardButton] = []
    if has_last:
        btns.append(InlineKeyboardButton("⟲ как в прошлый раз", callback_data=f"{prefix}:{CB_LAST}"))
    btns.append(InlineKeyboardButton("✏️ ввести своё", callback_data=f"{prefix}:{CB_CUSTOM}"))
    return btns


def hypotheses_kb(has_last: bool) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [_row(CB_HYP, [3, 5, DEFAULTS.hypotheses, 15]), _ending(CB_HYP, has_last)]
    )


def personas_kb(has_last: bool) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [_row(CB_PER, [5, 10, DEFAULTS.personas, 20]), _ending(CB_PER, has_last)]
    )


def turns_kb(has_last: bool) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [_row(CB_TURN, [20, 50, 80, DEFAULTS.max_turns]), _ending(CB_TURN, has_last)]
    )


def budget_kb(has_last: bool) -> InlineKeyboardMarkup:
    presets = [1_000_000, 3_000_000, DEFAULTS.token_budget, 10_000_000]
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton(f"{v // 1_000_000} млн", callback_data=f"{CB_BUDGET}:{v}") for v in presets],
            _ending(CB_BUDGET, has_last),
        ]
    )
