"""Хендлеры Telegram-бота: мастер настройки + приём файла + запуск прогона."""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationHandlerStop,
    CallbackQueryHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from .config import DEFAULTS, LIMITS, SQLITE_PATH, is_allowed
from .keyboards import (
    CB_BUDGET,
    CB_CUSTOM,
    CB_HYP,
    CB_LAST,
    CB_PER,
    CB_TURN,
    budget_kb,
    hypotheses_kb,
    personas_kb,
    turns_kb,
)
from .states import (
    KEY_HYPOTHESES,
    KEY_MAX_TURNS,
    KEY_PERSONAS,
    KEY_TOKEN_BUDGET,
    State,
)
from .storage import Storage

log = logging.getLogger(__name__)

# Тип callable, который оркестратор подставит на старте бота.
RunLauncher = Callable[
    [
        int,                                # user_id
        int,                                # run_id
        str,                                # file_path
        dict[str, Any],                     # params
        Callable[[str], Awaitable[None]],   # send_text
        Callable[[Path, str], Awaitable[None]],  # send_doc(path, caption)
    ],
    Awaitable[None],
]


# ---------- общие утилиты ----------------------------------------------------
async def _gate(update: Update) -> bool:
    """Whitelist по user_id. Возвращает True если пускаем дальше."""
    user = update.effective_user
    if user is None or not is_allowed(user.id):
        if update.effective_message:
            await update.effective_message.reply_text(
                "Доступ ограничен. Передай свой Telegram user_id владельцу бота."
            )
        return False
    return True


def _storage(context: ContextTypes.DEFAULT_TYPE) -> Storage:
    st = context.application.bot_data.get("storage")
    if st is None:
        st = Storage(SQLITE_PATH)
        context.application.bot_data["storage"] = st
    return st


def _set_state(context: ContextTypes.DEFAULT_TYPE, state: State) -> None:
    context.user_data["state"] = state


def _get_state(context: ContextTypes.DEFAULT_TYPE) -> State | None:
    return context.user_data.get("state")


def _get_param(context: ContextTypes.DEFAULT_TYPE, key: str) -> int | None:
    return context.user_data.get(key)


def _set_param(context: ContextTypes.DEFAULT_TYPE, key: str, value: int) -> None:
    context.user_data[key] = value


def _validate(value: int, lo: int, hi: int) -> int | None:
    return value if lo <= value <= hi else None


# ---------- /start -----------------------------------------------------------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _gate(update):
        return

    user = update.effective_user
    last = _storage(context).load_params(user.id) or {}
    context.user_data["last_params"] = last
    context.user_data.pop(KEY_HYPOTHESES, None)
    context.user_data.pop(KEY_PERSONAS, None)
    context.user_data.pop(KEY_MAX_TURNS, None)
    context.user_data.pop(KEY_TOKEN_BUDGET, None)

    await update.effective_message.reply_text(
        f"Привет, {user.first_name}. Настроим прогон.\n\n"
        f"<b>1/4. Сколько гипотез генерировать?</b>\n"
        f"Диапазон {LIMITS.hypotheses_min}–{LIMITS.hypotheses_max}, по умолчанию {DEFAULTS.hypotheses}.",
        parse_mode="HTML",
        reply_markup=hypotheses_kb(has_last=KEY_HYPOTHESES in last),
    )
    _set_state(context, State.ASK_HYPOTHESES)


# ---------- inline-кнопки ----------------------------------------------------
async def cb_hypotheses(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _handle_choice(
        update,
        context,
        prefix=CB_HYP,
        key=KEY_HYPOTHESES,
        lo=LIMITS.hypotheses_min,
        hi=LIMITS.hypotheses_max,
        next_state=State.ASK_PERSONAS,
        next_text=(
            f"<b>2/4. Сколько персон-клиентов на гипотезу?</b>\n"
            f"Диапазон {LIMITS.personas_min}–{LIMITS.personas_max}, по умолчанию {DEFAULTS.personas}."
        ),
        next_kb=lambda last: personas_kb(has_last=KEY_PERSONAS in last),
    )


async def cb_personas(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _handle_choice(
        update,
        context,
        prefix=CB_PER,
        key=KEY_PERSONAS,
        lo=LIMITS.personas_min,
        hi=LIMITS.personas_max,
        next_state=State.ASK_TURNS,
        next_text=(
            f"<b>3/4. Максимум реплик в одной симуляции?</b>\n"
            f"Диапазон {LIMITS.turns_min}–{LIMITS.turns_max}, по умолчанию {DEFAULTS.max_turns}.\n"
            f"<i>Реальная длина подстроится под медиану из загруженного файла.</i>"
        ),
        next_kb=lambda last: turns_kb(has_last=KEY_MAX_TURNS in last),
    )


async def cb_turns(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _handle_choice(
        update,
        context,
        prefix=CB_TURN,
        key=KEY_MAX_TURNS,
        lo=LIMITS.turns_min,
        hi=LIMITS.turns_max,
        next_state=State.ASK_BUDGET,
        next_text=(
            f"<b>4/4. Потолок по токенам на пачку?</b>\n"
            f"Если оценка превысит — прогон не стартует.\n"
            f"Диапазон {LIMITS.budget_min:,}–{LIMITS.budget_max:,}, "
            f"по умолчанию {DEFAULTS.token_budget:,}.".replace(",", " ")
        ),
        next_kb=lambda last: budget_kb(has_last=KEY_TOKEN_BUDGET in last),
    )


async def cb_budget(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    raw = query.data.split(":", 1)[1]
    last = context.user_data.get("last_params", {})

    if raw == CB_CUSTOM:
        await query.edit_message_text(
            f"Введи число от {LIMITS.budget_min:,} до {LIMITS.budget_max:,} "
            f"(потолок токенов).".replace(",", " ")
        )
        context.user_data["awaiting_custom_for"] = KEY_TOKEN_BUDGET
        return

    value = last.get(KEY_TOKEN_BUDGET) if raw == CB_LAST else int(raw)
    if value is None or _validate(value, LIMITS.budget_min, LIMITS.budget_max) is None:
        await query.edit_message_text("Не получилось. Введи число вручную.")
        context.user_data["awaiting_custom_for"] = KEY_TOKEN_BUDGET
        return

    _set_param(context, KEY_TOKEN_BUDGET, value)
    await query.edit_message_text(f"Потолок токенов: {value:,}".replace(",", " "))
    await _ask_for_file(update, context)


async def _handle_choice(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    *,
    prefix: str,
    key: str,
    lo: int,
    hi: int,
    next_state: State,
    next_text: str,
    next_kb,
) -> None:
    query = update.callback_query
    await query.answer()
    raw = query.data.split(":", 1)[1]
    last = context.user_data.get("last_params", {})

    if raw == CB_CUSTOM:
        await query.edit_message_text(f"Введи число от {lo} до {hi}.")
        context.user_data["awaiting_custom_for"] = key
        return

    value = last.get(key) if raw == CB_LAST else int(raw)
    if value is None or _validate(value, lo, hi) is None:
        await query.edit_message_text(f"Не получилось. Введи число от {lo} до {hi}.")
        context.user_data["awaiting_custom_for"] = key
        return

    _set_param(context, key, value)
    await query.edit_message_text(f"Принято: {value}")
    await update.effective_message.reply_text(
        next_text, parse_mode="HTML", reply_markup=next_kb(last)
    )
    _set_state(context, next_state)


# ---------- ввод вручную (текст) --------------------------------------------
async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _gate(update):
        return
    awaiting = context.user_data.get("awaiting_custom_for")
    if not awaiting:
        return  # игнор: не в режиме ввода

    text = (update.message.text or "").strip().replace(" ", "").replace("_", "")
    if not text.isdigit():
        await update.message.reply_text("Нужно положительное число.")
        return
    value = int(text)

    ranges = {
        KEY_HYPOTHESES: (LIMITS.hypotheses_min, LIMITS.hypotheses_max, State.ASK_PERSONAS),
        KEY_PERSONAS: (LIMITS.personas_min, LIMITS.personas_max, State.ASK_TURNS),
        KEY_MAX_TURNS: (LIMITS.turns_min, LIMITS.turns_max, State.ASK_BUDGET),
        KEY_TOKEN_BUDGET: (LIMITS.budget_min, LIMITS.budget_max, State.WAIT_FILE),
    }
    lo, hi, next_state = ranges[awaiting]
    if _validate(value, lo, hi) is None:
        await update.message.reply_text(f"Вне диапазона {lo}–{hi}. Попробуй ещё раз.")
        return

    _set_param(context, awaiting, value)
    context.user_data.pop("awaiting_custom_for", None)
    await update.message.reply_text(f"Принято: {value}")

    last = context.user_data.get("last_params", {})
    if next_state == State.ASK_PERSONAS:
        await update.message.reply_text(
            f"<b>2/4. Персоны?</b> {LIMITS.personas_min}–{LIMITS.personas_max}.",
            parse_mode="HTML",
            reply_markup=personas_kb(has_last=KEY_PERSONAS in last),
        )
    elif next_state == State.ASK_TURNS:
        await update.message.reply_text(
            f"<b>3/4. Реплики?</b> {LIMITS.turns_min}–{LIMITS.turns_max}.",
            parse_mode="HTML",
            reply_markup=turns_kb(has_last=KEY_MAX_TURNS in last),
        )
    elif next_state == State.ASK_BUDGET:
        await update.message.reply_text(
            f"<b>4/4. Потолок токенов?</b>",
            parse_mode="HTML",
            reply_markup=budget_kb(has_last=KEY_TOKEN_BUDGET in last),
        )
    else:
        await _ask_for_file(update, context)
    _set_state(context, next_state)


# ---------- запрос файла -----------------------------------------------------
async def _ask_for_file(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    p = context.user_data
    summary = (
        f"✅ Параметры:\n"
        f"• Гипотез: {p[KEY_HYPOTHESES]}\n"
        f"• Персон: {p[KEY_PERSONAS]}\n"
        f"• Макс. ходов: {p[KEY_MAX_TURNS]}\n"
        f"• Потолок токенов: {p[KEY_TOKEN_BUDGET]:,}".replace(",", " ")
    )
    await update.effective_message.reply_text(
        f"{summary}\n\n📎 Жду .xlsx с диалогами (до {LIMITS.file_size_mb} МБ)."
    )
    _set_state(context, State.WAIT_FILE)


# ---------- приём файла -----------------------------------------------------
async def on_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _gate(update):
        return
    if _get_state(context) != State.WAIT_FILE:
        await update.message.reply_text("Сначала /start — настроим параметры.")
        return

    doc = update.message.document
    if not doc:
        return
    if doc.file_size and doc.file_size > LIMITS.file_size_mb * 1024 * 1024:
        await update.message.reply_text(
            f"Файл больше {LIMITS.file_size_mb} МБ. Урежь и пришли заново."
        )
        return
    name = (doc.file_name or "").lower()
    if not name.endswith(".xlsx"):
        await update.message.reply_text("Нужен .xlsx (Excel).")
        return

    user_id = update.effective_user.id
    storage = _storage(context)
    if storage.has_active_run(user_id):
        await update.message.reply_text(
            "У тебя уже есть прогон в работе. Дождись окончания и пришли файл снова."
        )
        return

    save_dir = Path(".data/uploads") / str(user_id)
    save_dir.mkdir(parents=True, exist_ok=True)
    target = save_dir / f"{int(time.time())}_{doc.file_name}"

    await update.message.chat.send_action(ChatAction.TYPING)
    tg_file = await context.bot.get_file(doc.file_id)
    await tg_file.download_to_drive(custom_path=target)

    params = {
        KEY_HYPOTHESES: _get_param(context, KEY_HYPOTHESES),
        KEY_PERSONAS: _get_param(context, KEY_PERSONAS),
        KEY_MAX_TURNS: _get_param(context, KEY_MAX_TURNS),
        KEY_TOKEN_BUDGET: _get_param(context, KEY_TOKEN_BUDGET),
    }
    storage.save_params(user_id, params)
    run_id = storage.create_run(user_id, params, str(target))

    await update.message.reply_text(
        f"Принял файл: <code>{doc.file_name}</code>\n"
        f"ID прогона: <code>{run_id}</code>\nЗапускаю.",
        parse_mode="HTML",
    )
    _set_state(context, State.PROCESSING)

    launcher: RunLauncher | None = context.application.bot_data.get("run_launcher")
    if launcher is None:
        await update.message.reply_text(
            "⚠️ Оркестратор не подключён. Прогон зарегистрирован, но не стартовал."
        )
        storage.update_run_status(run_id, "failed", error="no run_launcher")
        return

    chat_id = update.effective_chat.id

    async def send_text(text: str) -> None:
        try:
            await context.bot.send_message(chat_id=chat_id, text=text, parse_mode="HTML")
        except Exception:
            log.exception("failed to send progress text")

    async def send_doc(path: Path, caption: str = "") -> None:
        try:
            with open(path, "rb") as f:
                await context.bot.send_document(
                    chat_id=chat_id,
                    document=f,
                    caption=caption or None,
                    filename=Path(path).name,
                )
        except Exception:
            log.exception("failed to send document %s", path)

    asyncio.create_task(
        _run_with_status(launcher, user_id, run_id, str(target), params, send_text, send_doc, storage)
    )


async def _run_with_status(
    launcher: RunLauncher,
    user_id: int,
    run_id: int,
    file_path: str,
    params: dict[str, Any],
    send_text: Callable[[str], Awaitable[None]],
    send_doc: Callable[[Path, str], Awaitable[None]],
    storage: Storage,
) -> None:
    storage.update_run_status(run_id, "running")
    try:
        await launcher(user_id, run_id, file_path, params, send_text, send_doc)
        storage.update_run_status(run_id, "done")
    except Exception as e:  # noqa: BLE001
        log.exception("run %s failed", run_id)
        storage.update_run_status(run_id, "failed", error=str(e))
        await send_text(f"❌ Прогон упал: {e}")


# ---------- регистрация ------------------------------------------------------
def build_handlers() -> list:
    from telegram.ext import CommandHandler

    return [
        CommandHandler("start", cmd_start),
        CallbackQueryHandler(cb_hypotheses, pattern=rf"^{CB_HYP}:"),
        CallbackQueryHandler(cb_personas, pattern=rf"^{CB_PER}:"),
        CallbackQueryHandler(cb_turns, pattern=rf"^{CB_TURN}:"),
        CallbackQueryHandler(cb_budget, pattern=rf"^{CB_BUDGET}:"),
        MessageHandler(filters.Document.ALL, on_document),
        MessageHandler(filters.TEXT & ~filters.COMMAND, on_text),
    ]
