"""RootCause's own tables: every run, every query (the audit trail) and user feedback."""

import json
import uuid
from datetime import datetime, timezone

from sqlalchemy import text

from backend.config import APP_SCHEMA, DEMO_MODE, admin_engine

DDL = f"""
CREATE SCHEMA IF NOT EXISTS {APP_SCHEMA};
CREATE TABLE IF NOT EXISTS {APP_SCHEMA}.runs (
    run_id       UUID PRIMARY KEY,
    question     TEXT NOT NULL,
    dataset      TEXT NOT NULL,
    status       TEXT NOT NULL,
    report       JSONB,
    usage        JSONB,
    created_at   TIMESTAMPTZ NOT NULL,
    duration_s   NUMERIC
);
CREATE TABLE IF NOT EXISTS {APP_SCHEMA}.queries (
    run_id        UUID REFERENCES {APP_SCHEMA}.runs(run_id) ON DELETE CASCADE,
    query_id      TEXT NOT NULL,
    purpose       TEXT,
    sql_submitted TEXT,
    sql_executed  TEXT,
    row_count     INTEGER,
    duration_ms   INTEGER,
    error         TEXT,
    preview       JSONB,
    PRIMARY KEY (run_id, query_id)
);
CREATE TABLE IF NOT EXISTS {APP_SCHEMA}.feedback (
    run_id     UUID REFERENCES {APP_SCHEMA}.runs(run_id) ON DELETE CASCADE,
    rating     SMALLINT NOT NULL,
    comment    TEXT,
    created_at TIMESTAMPTZ NOT NULL
);
"""


# SQLite (demo mode) has no schemas, JSONB or cross-database foreign keys; tables are attached as rootcause_app.
SQLITE_DDL = f"""
CREATE TABLE IF NOT EXISTS {APP_SCHEMA}.runs (
    run_id TEXT PRIMARY KEY, question TEXT NOT NULL, dataset TEXT NOT NULL, status TEXT NOT NULL,
    report TEXT, usage TEXT, created_at TEXT NOT NULL, duration_s REAL
);
CREATE TABLE IF NOT EXISTS {APP_SCHEMA}.queries (
    run_id TEXT, query_id TEXT NOT NULL, purpose TEXT, sql_submitted TEXT, sql_executed TEXT,
    row_count INTEGER, duration_ms INTEGER, error TEXT, preview TEXT, PRIMARY KEY (run_id, query_id)
);
CREATE TABLE IF NOT EXISTS {APP_SCHEMA}.feedback (
    run_id TEXT, rating INTEGER NOT NULL, comment TEXT, created_at TEXT NOT NULL
);
"""


def _json_param(name: str) -> str:
    return f":{name}" if DEMO_MODE else f"CAST(:{name} AS JSONB)"


def _decode(row: dict) -> dict:
    """SQLite returns JSON columns as text; PostgreSQL returns them parsed."""
    for key in ("report", "usage", "preview"):
        if isinstance(row.get(key), str):
            row[key] = json.loads(row[key])
    return row


def init_app_tables() -> None:
    with admin_engine().begin() as conn:
        for statement in filter(str.strip, (SQLITE_DDL if DEMO_MODE else DDL).split(";")):
            conn.execute(text(statement))


def save_run(result: dict) -> str:
    """Store a finished agent run (see backend.agent.runner) and return its id."""
    run_id = result.get("run_id") or str(uuid.uuid4())
    with admin_engine().begin() as conn:
        conn.execute(
            text(
                f"INSERT INTO {APP_SCHEMA}.runs (run_id, question, dataset, status, report, usage, created_at, duration_s) "
                f"VALUES (:run_id, :question, :dataset, :status, {_json_param('report')}, {_json_param('usage')}, :created_at, :duration_s)"
            ),
            {
                "run_id": run_id,
                "question": result["question"],
                "dataset": result["dataset"],
                "status": result["status"],
                "report": json.dumps(result.get("report"), default=str),
                "usage": json.dumps(result.get("usage"), default=str),
                "created_at": datetime.now(timezone.utc),
                "duration_s": result.get("duration_s"),
            },
        )
        for q in result.get("queries", []):
            conn.execute(
                text(
                    f"INSERT INTO {APP_SCHEMA}.queries (run_id, query_id, purpose, sql_submitted, sql_executed, row_count, duration_ms, error, preview) "
                    f"VALUES (:run_id, :query_id, :purpose, :sql_submitted, :sql_executed, :row_count, :duration_ms, :error, {_json_param('preview')})"
                ),
                {
                    "run_id": run_id,
                    "query_id": q["query_id"],
                    "purpose": q.get("purpose"),
                    "sql_submitted": q.get("sql_submitted"),
                    "sql_executed": q.get("sql_executed"),
                    "row_count": q.get("row_count"),
                    "duration_ms": q.get("duration_ms"),
                    "error": q.get("error"),
                    "preview": json.dumps(q.get("preview"), default=str),
                },
            )
    return run_id


def get_run(run_id: str) -> dict | None:
    with admin_engine().connect() as conn:
        row = conn.execute(
            text(f"SELECT * FROM {APP_SCHEMA}.runs WHERE run_id = :r"), {"r": run_id}
        ).mappings().first()
    return _decode(dict(row)) if row else None


def get_queries(run_id: str) -> list[dict]:
    with admin_engine().connect() as conn:
        rows = conn.execute(
            text(f"SELECT * FROM {APP_SCHEMA}.queries WHERE run_id = :r ORDER BY query_id"), {"r": run_id}
        ).mappings().all()
    return [_decode(dict(r)) for r in rows]


def list_runs(limit: int = 20) -> list[dict]:
    with admin_engine().connect() as conn:
        rows = conn.execute(
            text(
                f"SELECT run_id, question, dataset, status, created_at, duration_s FROM {APP_SCHEMA}.runs "
                "ORDER BY created_at DESC LIMIT :n"
            ),
            {"n": limit},
        ).mappings().all()
    return [dict(r) for r in rows]


def add_feedback(run_id: str, rating: int, comment: str | None) -> None:
    with admin_engine().begin() as conn:
        conn.execute(
            text(
                f"INSERT INTO {APP_SCHEMA}.feedback (run_id, rating, comment, created_at) "
                "VALUES (:r, :rating, :comment, :t)"
            ),
            {"r": run_id, "rating": rating, "comment": comment, "t": datetime.now(timezone.utc)},
        )
