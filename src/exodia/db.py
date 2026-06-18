"""SQLite store for community votes and per-idea feedback — the dynamic layer.

The pipeline owns the read-only content (ideas.json, KB, themes); this module
owns the interactive data the FastAPI app collects from visitors. There is one
vote per ``(idea_id, voter)`` token — a browser may change or retract its vote —
and feedback is append-only. The database is created on first use under the data
directory (``community.db``). A fresh connection is opened per call so the store
is safe to use from FastAPI's threadpool.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

from .util import now_utc_iso

_SCHEMA = """
CREATE TABLE IF NOT EXISTS votes (
  idea_id     TEXT NOT NULL,
  voter       TEXT NOT NULL,
  value       INTEGER NOT NULL CHECK (value IN (-1, 1)),
  created_utc TEXT NOT NULL,
  updated_utc TEXT NOT NULL,
  PRIMARY KEY (idea_id, voter)
);
CREATE TABLE IF NOT EXISTS feedback (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  idea_id     TEXT NOT NULL,
  author      TEXT,
  text        TEXT NOT NULL,
  created_utc TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_feedback_idea ON feedback(idea_id);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db(db_path: str | Path) -> None:
    with closing(connect(db_path)) as c:
        c.executescript(_SCHEMA)
        c.commit()


def _tally(conn: sqlite3.Connection, idea_id: str) -> dict:
    row = conn.execute(
        "SELECT COALESCE(SUM(value=1),0) up, COALESCE(SUM(value=-1),0) down "
        "FROM votes WHERE idea_id=?",
        (idea_id,),
    ).fetchone()
    up, down = int(row["up"]), int(row["down"])
    return {"idea_id": idea_id, "up": up, "down": down, "score": up - down}


def cast_vote(db_path: str | Path, idea_id: str, voter: str, value: int) -> dict:
    """Record a vote. ``value`` is -1, 0 (retract), or 1. Returns the idea's tally."""
    now = now_utc_iso()
    with closing(connect(db_path)) as c:
        if value == 0:
            c.execute("DELETE FROM votes WHERE idea_id=? AND voter=?", (idea_id, voter))
        else:
            c.execute(
                "INSERT INTO votes(idea_id,voter,value,created_utc,updated_utc) "
                "VALUES(?,?,?,?,?) ON CONFLICT(idea_id,voter) "
                "DO UPDATE SET value=excluded.value, updated_utc=excluded.updated_utc",
                (idea_id, voter, value, now, now),
            )
        c.commit()
        return _tally(c, idea_id)


def tallies(db_path: str | Path, idea_ids: list[str] | None = None) -> dict[str, dict]:
    """Vote tallies for all ideas; zero-filled for any ids in ``idea_ids``."""
    with closing(connect(db_path)) as c:
        rows = c.execute(
            "SELECT idea_id, COALESCE(SUM(value=1),0) up, COALESCE(SUM(value=-1),0) down "
            "FROM votes GROUP BY idea_id"
        ).fetchall()
    out = {
        r["idea_id"]: {"up": int(r["up"]), "down": int(r["down"]), "score": int(r["up"]) - int(r["down"])}
        for r in rows
    }
    for i in idea_ids or []:
        out.setdefault(i, {"up": 0, "down": 0, "score": 0})
    return out


def voter_votes(db_path: str | Path, voter: str) -> dict[str, int]:
    """Map of idea_id -> this voter's current vote (-1/1)."""
    with closing(connect(db_path)) as c:
        rows = c.execute("SELECT idea_id, value FROM votes WHERE voter=?", (voter,)).fetchall()
    return {r["idea_id"]: int(r["value"]) for r in rows}


def add_feedback(db_path: str | Path, idea_id: str, text: str, author: str | None = None) -> dict:
    now = now_utc_iso()
    with closing(connect(db_path)) as c:
        cur = c.execute(
            "INSERT INTO feedback(idea_id,author,text,created_utc) VALUES(?,?,?,?)",
            (idea_id, author, text, now),
        )
        c.commit()
        return {"id": cur.lastrowid, "idea_id": idea_id, "author": author,
                "text": text, "created_utc": now}


def list_feedback(db_path: str | Path, idea_id: str) -> list[dict]:
    with closing(connect(db_path)) as c:
        rows = c.execute(
            "SELECT id,idea_id,author,text,created_utc FROM feedback WHERE idea_id=? ORDER BY id",
            (idea_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def feedback_by_idea(db_path: str | Path) -> dict[str, list[dict]]:
    with closing(connect(db_path)) as c:
        rows = c.execute(
            "SELECT id,idea_id,author,text,created_utc FROM feedback ORDER BY id"
        ).fetchall()
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r["idea_id"], []).append(dict(r))
    return out
