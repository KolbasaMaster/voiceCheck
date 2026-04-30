"""Конфиг Telegram-бота: токен, whitelist, дефолтные параметры прогона."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class BotDefaults:
    hypotheses: int = 10
    personas: int = 15
    max_turns: int = 100
    token_budget: int = 5_000_000


@dataclass(frozen=True)
class BotLimits:
    hypotheses_min: int = 1
    hypotheses_max: int = 20
    personas_min: int = 1
    personas_max: int = 30
    turns_min: int = 4
    turns_max: int = 100
    budget_min: int = 100_000
    budget_max: int = 50_000_000
    file_size_mb: int = 30


def _parse_user_ids(raw: str | None) -> frozenset[int]:
    if not raw:
        return frozenset()
    out: set[int] = set()
    for chunk in raw.replace(";", ",").split(","):
        chunk = chunk.strip()
        if chunk and chunk.lstrip("-").isdigit():
            out.add(int(chunk))
    return frozenset(out)


TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
ALLOWED_USER_IDS: frozenset[int] = _parse_user_ids(os.getenv("TELEGRAM_ALLOWED_USER_IDS"))

# Идентификатор модели в config.yaml (см. src/text/__init__.py — register).
LLM_MODEL_ID: str = os.getenv("VOICECHECK_LLM_MODEL_ID", "gigachat_max")

# Пути относительно рабочей директории проекта (voiceCheck/voiceCheck/).
LLM_CONFIG_PATH: str = os.getenv("VOICECHECK_LLM_CONFIG", "config/config.yaml")
PROMPT_CONFIG_PATH: str = os.getenv("VOICECHECK_PROMPT_CONFIG", "config/prompt.yaml")
SQLITE_PATH: str = os.getenv("VOICECHECK_SQLITE", ".data/voicecheck.sqlite")

DEFAULTS = BotDefaults()
LIMITS = BotLimits()


def is_allowed(user_id: int) -> bool:
    """Whitelist по user_id. Пустой список = доступ запрещён всем."""
    return user_id in ALLOWED_USER_IDS
