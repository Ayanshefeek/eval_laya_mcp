"""
db.py

SQLite persistence for every judge call. One table, `eval_runs`, holding
enough to reconstruct what was judged, what Laya decided, and how
confident/fast it was -- without assuming every judge dimension produces
exactly one score.

Schema notes (see project discussion): `verdict` is technically derivable
from `scores` + `threshold` for four of the five judges (verdict = score >
threshold), but judge_injection collapses TWO scores (jailbreak,
prompt_injection) into ONE verdict via an OR, so verdict isn't always
recomputable from a single score column. `scores` is stored as a JSON blob
(not a flat REAL column) specifically so the schema doesn't assume "one
judge call = one score" -- it holds {"relevant": 0.57} for single-score
judges and {"jailbreak": 0.82, "prompt_injection": 0.86} for injection,
without needing a different table shape per dimension.

We open a short-lived connection per call rather than holding one open
connection module-wide: sqlite3 connections aren't safe to share across
threads by default, and an MCP server may field concurrent tool calls.
sqlite3.connect() is cheap enough that this isn't a meaningful cost per
judge call (which already takes ~1-2s for the Laya inference itself).
"""

import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Optional

from . import config

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS eval_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    source TEXT NOT NULL,
    judge_type TEXT NOT NULL,
    input_snapshot TEXT NOT NULL,
    verdict INTEGER NOT NULL,
    scores TEXT NOT NULL,
    threshold REAL NOT NULL,
    latency_ms REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_eval_runs_source ON eval_runs (source);
CREATE INDEX IF NOT EXISTS idx_eval_runs_judge_type ON eval_runs (judge_type);
CREATE INDEX IF NOT EXISTS idx_eval_runs_timestamp ON eval_runs (timestamp);
"""


_initialized = False


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(config.EVAL_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create the eval_runs table and its indexes if they don't exist yet.

    Safe to call more than once (CREATE TABLE/INDEX IF NOT EXISTS). Called
    explicitly by server.py on startup for clarity/logging, AND automatically
    by insert_eval_run/get_recent_runs the first time either is used --
    judges.py is meant to be importable and usable standalone (see its
    module docstring), without requiring the caller to have run server.py
    first, so the table can't only be created there.
    """
    global _initialized
    with _get_connection() as conn:
        conn.executescript(_SCHEMA)
    _initialized = True
    logger.info("Eval DB ready at %s", config.EVAL_DB_PATH)


def _ensure_initialized() -> None:
    if not _initialized:
        init_db()


def insert_eval_run(
    source: str,
    judge_type: str,
    input_snapshot: dict,
    verdict: bool,
    scores: dict,
    threshold: float,
    latency_ms: float,
) -> int:
    """Insert one judge call's result. Returns the new row's id."""
    _ensure_initialized()
    with _get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO eval_runs
                (timestamp, source, judge_type, input_snapshot, verdict, scores, threshold, latency_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                source,
                judge_type,
                json.dumps(input_snapshot),
                int(verdict),
                json.dumps(scores),
                threshold,
                latency_ms,
            ),
        )
        return cursor.lastrowid


def get_recent_runs(limit: int = 20, source: Optional[str] = None, judge_type: Optional[str] = None) -> list:
    """Return the most recent eval_runs rows, newest first, as plain dicts.

    This is a lightweight read helper for sanity-checking Phase 3 -- Phase 4
    adds real aggregation (pass rates, averages) on top of this table.
    """
    _ensure_initialized()
    query = "SELECT * FROM eval_runs"
    conditions = []
    params: list = []
    if source is not None:
        conditions.append("source = ?")
        params.append(source)
    if judge_type is not None:
        conditions.append("judge_type = ?")
        params.append(judge_type)
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    with _get_connection() as conn:
        rows = conn.execute(query, params).fetchall()

    results = []
    for row in rows:
        d = dict(row)
        d["input_snapshot"] = json.loads(d["input_snapshot"])
        d["scores"] = json.loads(d["scores"])
        d["verdict"] = bool(d["verdict"])
        results.append(d)
    return results


def get_eval_summary(
    source: Optional[str] = None,
    judge_type: Optional[str] = None,
    since_days: Optional[int] = None,
) -> dict:
    """Aggregate eval_runs into pass rate / average score / average latency,
    grouped by judge_type. `since_days` restricts to runs from the last N
    days (None = all time).

    Scores are aggregated in Python rather than via SQLite's JSON1 functions
    (json_extract etc.) -- eval log volumes here are small enough that this
    is simpler and doesn't depend on JSON1 being compiled into whatever
    Python/sqlite3 build a given machine ships, which isn't guaranteed
    across platforms.

    `avg_score` averages every individual score value stored for that
    judge_type -- for judge_injection, which logs two scores per row
    (jailbreak, prompt_injection), this means each contributes its own
    value to the average rather than one row contributing one number.
    `pass_rate` uses `verdict` directly, so it's unaffected by that:
    injection's verdict is already the OR of both sub-scores per row.

    Returns:
        {
          "filters": {"source": ..., "judge_type": ..., "since_days": ...},
          "total_runs": <int>,
          "by_judge_type": {
            "<judge_type>": {
              "count": <int>, "pass_rate": <float|None>,
              "avg_score": <float|None>, "avg_latency_ms": <float|None>
            }, ...
          }
        }
    """
    _ensure_initialized()
    query = "SELECT judge_type, verdict, scores, latency_ms, timestamp FROM eval_runs"
    conditions = []
    params: list = []
    if source is not None:
        conditions.append("source = ?")
        params.append(source)
    if judge_type is not None:
        conditions.append("judge_type = ?")
        params.append(judge_type)
    if since_days is not None:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=since_days)).isoformat()
        conditions.append("timestamp >= ?")
        params.append(cutoff)
    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    with _get_connection() as conn:
        rows = conn.execute(query, params).fetchall()

    buckets: dict = {}
    for row in rows:
        jt = row["judge_type"]
        b = buckets.setdefault(jt, {"count": 0, "pass_count": 0, "score_sum": 0.0, "score_n": 0, "latency_sum": 0.0})
        b["count"] += 1
        b["pass_count"] += int(row["verdict"])
        b["latency_sum"] += row["latency_ms"]
        for value in json.loads(row["scores"]).values():
            b["score_sum"] += value
            b["score_n"] += 1

    by_judge_type = {}
    for jt, b in buckets.items():
        by_judge_type[jt] = {
            "count": b["count"],
            "pass_rate": round(b["pass_count"] / b["count"], 4) if b["count"] else None,
            "avg_score": round(b["score_sum"] / b["score_n"], 4) if b["score_n"] else None,
            "avg_latency_ms": round(b["latency_sum"] / b["count"], 2) if b["count"] else None,
        }

    return {
        "filters": {"source": source, "judge_type": judge_type, "since_days": since_days},
        "total_runs": len(rows),
        "by_judge_type": by_judge_type,
    }