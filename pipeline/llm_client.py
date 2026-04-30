"""Тонкий async-клиент над `GigaChatModel`: рендер промптов + вызовы в to_thread.

Существующий `LLMOrchestrator` (src/text/llm.py) заточен под аудио-пайплайн
Callytics (`Formatter.format_ssm_as_dialogue`), поэтому для бота используем
свой клиент — гибче, без блокировок event loop.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import yaml

# Регистрация моделей в ModelRegistry — побочный эффект импорта.
from src.text import ModelRegistry  # noqa: F401  (импорт инициализирует реестр)
from src.text.model import ModelFactory


log = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.+?)\s*```", re.DOTALL | re.IGNORECASE)
_JSON_DECODER = json.JSONDecoder()


def _resolve_env_vars(d: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in d.items():
        if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
            out[k] = os.getenv(v[2:-1], "")
        else:
            out[k] = v
    return out


def extract_json(text: str) -> dict[str, Any] | None:
    """Достаём первый валидный JSON-объект (поддерживает ```json``` fences и произвольную вложенность)."""
    if not text:
        return None
    # 1) Чистый JSON целиком.
    stripped = text.strip()
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    # 2) JSON внутри ```json ... ``` или ``` ... ```.
    m = _FENCE_RE.search(text)
    if m:
        candidate = m.group(1).strip()
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    # 3) Идём по каждой '{' и пробуем raw_decode — поддерживает любую вложенность.
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _end = _JSON_DECODER.raw_decode(text, i)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


class LLMClient:
    """Async-обёртка над одной моделью + загруженные промпты."""

    def __init__(
        self,
        *,
        model_id: str,
        config_path: str,
        prompt_path: str,
    ) -> None:
        self.model_id = model_id
        with open(config_path, encoding="utf-8") as f:
            full_config = yaml.safe_load(f) or {}
        models = full_config.get("models", {})
        if model_id not in models:
            raise ValueError(f"В {config_path} не найден раздел models.{model_id}")
        model_config = _resolve_env_vars(models[model_id])
        model_config["compute_type"] = full_config.get("runtime", {}).get(
            "compute_type", "float16"
        )
        self.model = ModelFactory.create_model(model_id, model_config)

        with open(prompt_path, encoding="utf-8") as f:
            self.prompts: dict[str, dict[str, str]] = yaml.safe_load(f) or {}

        # Серилизуем доступ к клиенту (gigachat-SDK не обещает thread-safety).
        self._sem = asyncio.Semaphore(1)

        # Лёгкий счётчик использования — для отчёта.
        self.calls = 0
        self.last_error: str | None = None

    def render(self, prompt_name: str, **vars: Any) -> tuple[str, str]:
        spec = self.prompts.get(prompt_name)
        if not spec:
            raise KeyError(f"prompt '{prompt_name}' не найден в prompt.yaml")
        sys_t = spec.get("system", "")
        usr_t = spec.get("user", "")
        return sys_t.format(**vars), usr_t.format(**vars)

    async def chat(
        self,
        *,
        system: str,
        user: str,
        max_length: int = 4000,
        temperature: float = 0.7,
        retries: int = 2,
    ) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        last_exc: Exception | None = None
        for attempt in range(retries + 1):
            async with self._sem:
                try:
                    self.calls += 1
                    return await asyncio.to_thread(
                        self.model.generate,
                        messages,
                        max_length=max_length,
                        temperature=temperature,
                    )
                except Exception as e:  # noqa: BLE001
                    last_exc = e
                    self.last_error = str(e)
                    log.warning("LLM call failed (attempt %s/%s): %s", attempt + 1, retries + 1, e)
            await asyncio.sleep(min(8.0, 1.5 ** attempt))
        raise RuntimeError(f"LLM call failed after {retries + 1} attempts: {last_exc}")

    async def chat_json(
        self,
        *,
        system: str,
        user: str,
        max_length: int = 4000,
        temperature: float = 0.3,
        retries: int = 2,
    ) -> dict[str, Any] | None:
        text = await self.chat(
            system=system, user=user, max_length=max_length,
            temperature=temperature, retries=retries,
        )
        data = extract_json(text)
        if data is None:
            log.warning("chat_json: не удалось распарсить JSON. Сырой ответ (первые 800 симв.):\n%s", (text or "")[:800])
        return data

    def close(self) -> None:
        try:
            self.model.unload()
        except Exception:  # noqa: BLE001
            pass
