"""Стейты FSM мастера настройки прогона."""
from __future__ import annotations

from enum import IntEnum, auto


class State(IntEnum):
    ASK_HYPOTHESES = auto()
    ASK_PERSONAS = auto()
    ASK_TURNS = auto()
    ASK_BUDGET = auto()
    WAIT_FILE = auto()
    PROCESSING = auto()


# Ключи для context.user_data / БД.
KEY_HYPOTHESES = "hypotheses"
KEY_PERSONAS = "personas"
KEY_MAX_TURNS = "max_turns"
KEY_TOKEN_BUDGET = "token_budget"

PARAM_KEYS = (KEY_HYPOTHESES, KEY_PERSONAS, KEY_MAX_TURNS, KEY_TOKEN_BUDGET)
