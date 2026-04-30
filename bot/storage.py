"""SQLite-хранилище: последние параметры пользователя + история прогонов + очередь.

Один пользователь = одна одновременная очередь прогонов.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

from .states import PARAM_KEYS


_SCHEMA = """
CREATE TABLE IF NOT EXISTS user_settings (
    user_id      INTEGER PRIMARY KEY,
    params_json  TEXT NOT NULL,
    updated_at   REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      INTEGER NOT NULL,
    status       TEXT NOT NULL,           -- queued | running | done | failed | cancelled
    params_json  TEXT NOT NULL,
    file_path    TEXT,
    started_at   REAL NOT NULL,
    finished_at  REAL,
    error        TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_user_status ON runs(user_id, status);
"""


class Storage:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # --- последние параметры ------------------------------------------------
    def save_params(self, user_id: int, params: dict[str, Any]) -> None:
        payload = json.dumps({k: params[k] for k in PARAM_KEYS if k in params})
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO user_settings(user_id, params_json, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(user_id) DO UPDATE SET params_json=excluded.params_json, "
                "updated_at=excluded.updated_at",
                (user_id, payload, time.time()),
            )

    def load_params(self, user_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT params_json FROM user_settings WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        if not row:
            return None
        return json.loads(row["params_json"])

    # --- очередь прогонов ---------------------------------------------------
    def has_active_run(self, user_id: int) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM runs WHERE user_id = ? AND status IN ('queued','running') LIMIT 1",
                (user_id,),
            ).fetchone()
        return row is not None

    def create_run(self, user_id: int, params: dict[str, Any], file_path: str) -> int:
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO runs(user_id, status, params_json, file_path, started_at) "
                "VALUES (?, 'queued', ?, ?, ?)",
                (user_id, json.dumps(params), file_path, time.time()),
            )
            return int(cur.lastrowid)

    def update_run_status(
        self,
        run_id: int,
        status: str,
        error: str | None = None,
    ) -> None:
        finished = time.time() if status in {"done", "failed", "cancelled"} else None
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE runs SET status = ?, finished_at = COALESCE(?, finished_at), error = ? "
                "WHERE id = ?",
                (status, finished, error, run_id),
            )

    def list_recent_runs(self, user_id: int, limit: int = 5) -> list[sqlite3.Row]:
        with self._connect() as conn:
            return list(
                conn.execute(
                    "SELECT id, status, started_at, finished_at, error FROM runs "
                    "WHERE user_id = ? ORDER BY id DESC LIMIT ?",
                    (user_id, limit),
                )
            )
