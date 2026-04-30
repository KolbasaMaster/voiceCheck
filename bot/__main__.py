"""Точка входа Telegram-бота: `python -m bot`."""
from __future__ import annotations

import logging

from telegram.ext import Application, ApplicationBuilder

from .config import ALLOWED_USER_IDS, SQLITE_PATH, TELEGRAM_BOT_TOKEN
from .handlers import build_handlers
from .storage import Storage


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("telegram").setLevel(logging.INFO)


def main() -> None:
    _setup_logging()
    log = logging.getLogger("bot")

    if not TELEGRAM_BOT_TOKEN:
        raise SystemExit("TELEGRAM_BOT_TOKEN не задан в .env")
    if not ALLOWED_USER_IDS:
        log.warning(
            "TELEGRAM_ALLOWED_USER_IDS пуст — все запросы будут отклонены. "
            "Добавь свой user_id (узнать у @userinfobot)."
        )

    app: Application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.bot_data["storage"] = Storage(SQLITE_PATH)

    # Подключаем оркестратор. Импорт ленивый — модель тяжёлая, не хочется
    # тащить её при простых проверках конфига.
    from pipeline.orchestrator import build_run_launcher

    app.bot_data["run_launcher"] = build_run_launcher()

    for handler in build_handlers():
        app.add_handler(handler)

    log.info("Бот запущен. Whitelist: %s", sorted(ALLOWED_USER_IDS) or "пусто")
    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()
