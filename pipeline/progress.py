"""Канал отчёта о прогрессе: оркестратор → пользователь Telegram."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Protocol


class ProgressReporter(Protocol):
    async def text(self, msg: str) -> None: ...
    async def document(self, path: Path, caption: str = "") -> None: ...


@dataclass
class CallableReporter:
    """Простая реализация поверх двух callable'ов."""

    send_text: Callable[[str], Awaitable[None]]
    send_doc: Callable[[Path, str], Awaitable[None]]

    async def text(self, msg: str) -> None:
        await self.send_text(msg)

    async def document(self, path: Path, caption: str = "") -> None:
        await self.send_doc(path, caption)
