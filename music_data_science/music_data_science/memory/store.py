"""SQLite system-of-record for sessions, stems, generations, and scores.

Uses only the stdlib ``sqlite3`` module.  This is the durable memory layer:
every generation request/result pair and every Best-of-N score lands here so
sessions can be replayed, audited, and mined for telemetry.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from typing import Any, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL DEFAULT '',
    blueprint_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS stems (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    stem_class TEXT NOT NULL,
    caption TEXT NOT NULL DEFAULT '',
    audio_path TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS generations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    task_id TEXT NOT NULL,
    task_type TEXT NOT NULL,
    seed INTEGER,
    request_json TEXT NOT NULL DEFAULT '{}',
    result_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    generation_id INTEGER NOT NULL REFERENCES generations(id),
    total REAL NOT NULL,
    accepted INTEGER NOT NULL DEFAULT 0,
    breakdown_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL
);
"""


class MemoryStore:
    """Durable store for the orchestration workflow.

    Args:
        path: SQLite database path; use ``":memory:"`` for an ephemeral store.
    """

    def __init__(self, path: str = ":memory:") -> None:
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def create_session(self, name: str = "", blueprint: Optional[dict[str, Any]] = None) -> str:
        """Create a session row and return its generated ID."""
        session_id = uuid.uuid4().hex
        self._conn.execute(
            "INSERT INTO sessions (id, name, blueprint_json, created_at) VALUES (?, ?, ?, ?)",
            (session_id, name, json.dumps(blueprint or {}), time.time()),
        )
        self._conn.commit()
        return session_id

    def get_session(self, session_id: str) -> Optional[dict[str, Any]]:
        """Return one session as a dict, or ``None`` when missing."""
        row = self._conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return dict(row) if row else None

    def add_stem(self, session_id: str, stem_class: str, caption: str = "", audio_path: str = "") -> int:
        """Record a stem for a session; return the stem row ID."""
        cursor = self._conn.execute(
            "INSERT INTO stems (session_id, stem_class, caption, audio_path, created_at) VALUES (?, ?, ?, ?, ?)",
            (session_id, stem_class, caption, audio_path, time.time()),
        )
        self._conn.commit()
        return int(cursor.lastrowid or 0)

    def list_stems(self, session_id: str) -> list[dict[str, Any]]:
        """Return all stems for a session, oldest first."""
        rows = self._conn.execute(
            "SELECT * FROM stems WHERE session_id = ? ORDER BY id", (session_id,)
        ).fetchall()
        return [dict(row) for row in rows]

    def record_generation(
        self,
        session_id: str,
        task_id: str,
        task_type: str,
        request: dict[str, Any],
        result: dict[str, Any],
        seed: Optional[int] = None,
    ) -> int:
        """Persist one generation request/result pair; return the row ID."""
        cursor = self._conn.execute(
            "INSERT INTO generations (session_id, task_id, task_type, seed, request_json, result_json,"
            " created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (session_id, task_id, task_type, seed, json.dumps(request), json.dumps(result), time.time()),
        )
        self._conn.commit()
        return int(cursor.lastrowid or 0)

    def list_generations(self, session_id: str) -> list[dict[str, Any]]:
        """Return all generations for a session, oldest first, with decoded JSON."""
        rows = self._conn.execute(
            "SELECT * FROM generations WHERE session_id = ? ORDER BY id", (session_id,)
        ).fetchall()
        return [self._decode_generation(row) for row in rows]

    @staticmethod
    def _decode_generation(row: sqlite3.Row) -> dict[str, Any]:
        """Decode a generation row's JSON columns into dicts."""
        record = dict(row)
        record["request"] = json.loads(record.pop("request_json"))
        record["result"] = json.loads(record.pop("result_json"))
        return record

    def record_score(self, generation_id: int, total: float, accepted: bool, breakdown: dict[str, float]) -> int:
        """Persist one Best-of-N score for a generation; return the row ID."""
        cursor = self._conn.execute(
            "INSERT INTO scores (generation_id, total, accepted, breakdown_json, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (generation_id, total, int(accepted), json.dumps(breakdown), time.time()),
        )
        self._conn.commit()
        return int(cursor.lastrowid or 0)

    def best_generation(self, session_id: str) -> Optional[dict[str, Any]]:
        """Return the highest-scored generation for a session, or ``None``."""
        row = self._conn.execute(
            "SELECT g.*, s.total AS score_total FROM generations g"
            " JOIN scores s ON s.generation_id = g.id"
            " WHERE g.session_id = ? ORDER BY s.total DESC, g.id ASC LIMIT 1",
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        record = self._decode_generation(row)
        return record
