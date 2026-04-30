"""Оркестратор: связывает препроцессинг, генератор гипотез, симуляцию, цензор и репортёр.

`build_run_launcher()` возвращает callable, который handlers.py подключает в
`app.bot_data["run_launcher"]`. Внутри одного процесса бота переиспользуется
один и тот же `LLMClient` — модель GigaChat-Max инициализируется единожды.
"""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

from bot.config import LLM_CONFIG_PATH, LLM_MODEL_ID, PROMPT_CONFIG_PATH
from bot.states import KEY_HYPOTHESES, KEY_MAX_TURNS, KEY_PERSONAS, KEY_TOKEN_BUDGET

from .agents.censor import evaluate as censor_eval
from .agents.hypothesis_generator import Hypothesis, generate_hypotheses
from .agents.persona_extractor import Persona, extract_personas
from .agents.reporter import HypothesisReport, report_hypothesis
from .agents.simulator import SimResult, simulate
from .budget import estimate
from .export import export_run
from .llm_client import LLMClient
from .preprocess import (
    Dialogue,
    adaptive_turn_cap,
    compute_stats,
    mark_outcomes,
    parse_xlsx,
    sample_balanced,
)


log = logging.getLogger(__name__)

# Сколько диалогов из корпуса берём в LLM-сэмпл (для генератора гипотез/персон).
SAMPLE_FOR_LLM = 60
# Сколько прогонов симуляции крутим параллельно (одновременно к одному
# GigaChat-клиенту). Реально серилизуется семафором внутри LLMClient,
# но малый параллелизм даёт лучшую утилизацию из-за сетевых пауз.
SIM_CONCURRENCY = 4


# --------------------------------------------------------------------------
#  Reporter — ленивая отправка прогресса в чат
# --------------------------------------------------------------------------
class _Progress:
    def __init__(self, send_text, send_doc):
        self._t = send_text
        self._d = send_doc

    async def text(self, msg: str) -> None:
        await self._t(msg)

    async def document(self, path: Path, caption: str = "") -> None:
        await self._d(path, caption)


# --------------------------------------------------------------------------
#  Глобальный синглтон LLM-клиента
# --------------------------------------------------------------------------
_llm_lock = asyncio.Lock()
_llm: LLMClient | None = None


async def _get_llm() -> LLMClient:
    global _llm
    async with _llm_lock:
        if _llm is None:
            _llm = LLMClient(
                model_id=LLM_MODEL_ID,
                config_path=LLM_CONFIG_PATH,
                prompt_path=PROMPT_CONFIG_PATH,
            )
        return _llm


# --------------------------------------------------------------------------
#  Основной пайплайн
# --------------------------------------------------------------------------
async def run_pipeline(
    *,
    user_id: int,
    run_id: int,
    file_path: str,
    params: dict[str, Any],
    progress: _Progress,
) -> None:
    n_hyp = int(params[KEY_HYPOTHESES])
    n_per = int(params[KEY_PERSONAS])
    user_turn_cap = int(params[KEY_MAX_TURNS])
    token_budget = int(params[KEY_TOKEN_BUDGET])

    started = time.time()
    await progress.text("📂 Парсю xlsx…")
    dialogues: list[Dialogue] = await asyncio.to_thread(parse_xlsx, file_path)
    if not dialogues:
        await progress.text("❌ Не нашёл диалогов в файле.")
        return

    mark_outcomes(dialogues)
    stats = compute_stats(dialogues)
    turn_cap = adaptive_turn_cap(stats, user_turn_cap)

    outcome_summary = ", ".join(f"{k}={v}" for k, v in sorted(stats.outcome_counts.items()))
    await progress.text(
        f"📊 Корпус: {stats.dialogue_count} диалогов, медиана {stats.median_turns} реплик, "
        f"p90 {stats.p90_turns}.\n"
        f"Эвристика исходов: {outcome_summary}\n"
        f"Адаптивный cap по ходам в симуляции: {turn_cap}."
    )

    # --- бюджет --------------------------------------------------------------
    sample_size = min(SAMPLE_FOR_LLM, stats.dialogue_count)
    est = estimate(
        n_dialogues_sample=sample_size,
        avg_turn_chars=stats.avg_chars_per_turn,
        n_hypotheses=n_hyp,
        n_personas=n_per,
        n_turns=turn_cap,
    )
    await progress.text(f"💰 Оценка бюджета:\n<code>{est.as_table()}</code>")
    if est.total > token_budget:
        await progress.text(
            f"🛑 Оценка ({est.total:,}) больше пользовательского потолка ({token_budget:,}). "
            f"Прогон не запускаю — урежь параметры и попробуй снова.".replace(",", " ")
        )
        return

    # --- сэмпл для LLM-этапов -----------------------------------------------
    # seed = run_id, чтобы разные прогоны брали разные подмножества корпуса,
    # но один и тот же run воспроизводился (для дебага/повторов).
    sample = sample_balanced(dialogues, n=sample_size, seed=run_id)
    llm = await _get_llm()

    await progress.text("🧠 Генерирую гипотезы и извлекаю персоны…")
    hyp_task = asyncio.create_task(generate_hypotheses(llm, sample, n_hyp))
    per_task = asyncio.create_task(extract_personas(llm, sample, n_per))
    hypotheses, personas = await asyncio.gather(hyp_task, per_task)

    if not hypotheses:
        await progress.text("❌ Не получилось сгенерировать гипотезы. Прерываю.")
        return
    if not personas:
        await progress.text("❌ Не получилось извлечь персоны. Прерываю.")
        return

    await progress.text(
        f"✅ Готово: {len(hypotheses)} гипотез × {len(personas)} персон "
        f"= {len(hypotheses) * len(personas)} прогонов симуляции."
    )

    # --- симуляция -----------------------------------------------------------
    sims: list[SimResult] = []
    sem = asyncio.Semaphore(SIM_CONCURRENCY)
    total = len(hypotheses) * len(personas)
    done_counter = {"n": 0}
    last_ping = {"t": time.time()}

    async def _one(h: Hypothesis, p: Persona) -> SimResult:
        async with sem:
            res = await simulate(llm, h, p, max_turns=turn_cap)
        done_counter["n"] += 1
        # Не спамим — апдейт раз в ~30 сек или каждые 10% прогонов.
        n = done_counter["n"]
        if n == total or time.time() - last_ping["t"] > 30 or n % max(1, total // 10) == 0:
            last_ping["t"] = time.time()
            await progress.text(f"🎭 Симуляция: {n}/{total}")
        return res

    sim_tasks = [_one(h, p) for h in hypotheses for p in personas]
    for fut in asyncio.as_completed(sim_tasks):
        sims.append(await fut)

    # --- цензор --------------------------------------------------------------
    await progress.text("⚖️ Цензор оценивает прогоны…")
    by_pair: dict[tuple[str, str], SimResult] = {(s.hypothesis_id, s.persona_id): s for s in sims}
    hyp_by_id = {h.id: h for h in hypotheses}
    per_by_id = {p.id: p for p in personas}

    censor_sem = asyncio.Semaphore(SIM_CONCURRENCY)

    async def _ev(sim: SimResult):
        h = hyp_by_id[sim.hypothesis_id]
        p = per_by_id[sim.persona_id]
        async with censor_sem:
            return await censor_eval(llm, h, p, sim)

    verdicts = await asyncio.gather(*[_ev(s) for s in sims])

    # --- репортёр ------------------------------------------------------------
    await progress.text("📝 Репортёр собирает саммари по гипотезам…")
    by_hyp: dict[str, list] = {h.id: [] for h in hypotheses}
    for v in verdicts:
        by_hyp.setdefault(v.hypothesis_id, []).append(v)

    reports: list[HypothesisReport] = []
    for h in hypotheses:
        rep = await report_hypothesis(llm, h, by_hyp.get(h.id, []))
        reports.append(rep)

    # --- экспорт + финальное сообщение --------------------------------------
    elapsed = int(time.time() - started)
    out_dir = Path(".data/reports") / str(user_id)
    out_path = out_dir / f"run_{run_id}.xlsx"
    meta = {
        "run_id": run_id,
        "user_id": user_id,
        "elapsed_sec": elapsed,
        "n_hypotheses": n_hyp,
        "n_personas": n_per,
        "max_turns": turn_cap,
        "token_budget": token_budget,
        "estimated_tokens": est.total,
        "llm_calls": llm.calls,
        "corpus_size": stats.dialogue_count,
    }
    await asyncio.to_thread(
        export_run,
        out_path=out_path,
        hypotheses=hypotheses,
        personas=personas,
        sims=sims,
        verdicts=verdicts,
        reports=reports,
        meta=meta,
    )

    # Текстовая выжимка прямо в чат.
    lines = [f"✅ Прогон #{run_id} закончен за {elapsed} сек.\n"]
    for r in reports:
        h = hyp_by_id.get(r.hypothesis_id)
        title = f"{r.hypothesis_id}. {h.name if h else r.hypothesis_id} ({h.methodology if h else '—'})"
        lines.append(
            f"<b>{title}</b>\n"
            f"  Вердикт: {r.verdict}, средний балл: {r.overall_score}\n"
            f"  Лучшая персона: {r.best_persona}, худшая: {r.worst_persona}"
        )
    await progress.text("\n\n".join(lines))
    await progress.document(out_path, caption=f"Полный отчёт по прогону #{run_id}")


# --------------------------------------------------------------------------
#  Адаптер для handlers.py
# --------------------------------------------------------------------------
def build_run_launcher() -> Callable[..., Awaitable[None]]:
    async def launcher(
        user_id: int,
        run_id: int,
        file_path: str,
        params: dict[str, Any],
        send_text: Callable[[str], Awaitable[None]],
        send_doc: Callable[[Path, str], Awaitable[None]] | None = None,
    ) -> None:
        # Совместимость со старой сигнатурой (только текст): если send_doc не
        # передан — отправляем текстом путь к файлу.
        if send_doc is None:
            async def _stub(path: Path, caption: str = "") -> None:
                await send_text(f"📄 Отчёт сохранён: <code>{path}</code>\n{caption}")
            send_doc = _stub
        progress = _Progress(send_text=send_text, send_doc=send_doc)
        await run_pipeline(
            user_id=user_id,
            run_id=run_id,
            file_path=file_path,
            params=params,
            progress=progress,
        )

    return launcher
