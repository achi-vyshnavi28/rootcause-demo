"""RootCause's own tables: every run, every query (the audit trail) and user feedback.

The audit_log table makes the trail tamper-evident (21 CFR Part 11 §11.10(e), ALCOA+): every saved run and every
feedback appends an entry holding a SHA-256 digest of the record and the hash of the previous entry. verify_audit()
recomputes the chain and re-reads each run from the tables, so an edit made directly in the database is reported.
"""

import hashlib
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
CREATE TABLE IF NOT EXISTS {APP_SCHEMA}.audit_log (
    seq        BIGINT PRIMARY KEY,
    event      TEXT NOT NULL,
    run_id     TEXT NOT NULL,
    payload    TEXT NOT NULL,
    at         TEXT NOT NULL,
    prev_hash  TEXT NOT NULL,
    hash       TEXT NOT NULL
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
CREATE TABLE IF NOT EXISTS {APP_SCHEMA}.audit_log (
    seq INTEGER PRIMARY KEY, event TEXT NOT NULL, run_id TEXT NOT NULL, payload TEXT NOT NULL, at TEXT NOT NULL,
    prev_hash TEXT NOT NULL, hash TEXT NOT NULL
);
"""

GENESIS = "0" * 64


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
        _baseline_legacy_runs(conn)


def _baseline_legacy_runs(conn) -> None:
    """Runs saved before the audit log existed are baselined once (their state at go-live), then protected."""
    logged = {r[0] for r in conn.execute(text(f"SELECT run_id FROM {APP_SCHEMA}.audit_log")).all()}
    legacy = conn.execute(text(f"SELECT run_id FROM {APP_SCHEMA}.runs ORDER BY created_at")).all()
    for (run_id,) in legacy:
        if str(run_id) not in logged:
            snap = _run_snapshot(conn, str(run_id))
            _append_audit(conn, "run_baselined", str(run_id), {
                "digest": _digest(snap), "question": snap["question"], "status": snap["status"],
                "models": snap["models"], "queries": len(snap["queries"]),
            })


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
        snap = _run_snapshot(conn, run_id)
        _append_audit(conn, "run_recorded", run_id, {
            "digest": _digest(snap), "question": snap["question"], "status": snap["status"],
            "models": snap["models"], "queries": len(snap["queries"]),
        })
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
        _append_audit(conn, "feedback", run_id, {"rating": rating, "comment": comment})


# --- tamper-evident audit log -------------------------------------------------------------------------

def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))


def _digest(obj) -> str:
    return hashlib.sha256(_canonical(obj).encode()).hexdigest()


def _run_snapshot(conn, run_id: str) -> dict:
    """What an auditor must be able to trust about a run, read back from the tables as stored."""
    run = _decode(dict(conn.execute(text(f"SELECT * FROM {APP_SCHEMA}.runs WHERE run_id = :r"),
                                    {"r": run_id}).mappings().one()))
    queries = conn.execute(
        text(f"SELECT query_id, sql_executed, row_count, error FROM {APP_SCHEMA}.queries WHERE run_id = :r "
             "ORDER BY query_id"), {"r": run_id}).mappings().all()
    return {
        "run_id": str(run["run_id"]), "question": run["question"], "dataset": run["dataset"], "status": run["status"],
        "report": run["report"], "models": (run.get("usage") or {}).get("models", []),
        "created_at": str(run["created_at"]), "queries": [dict(q) for q in queries],
    }


def _append_audit(conn, event: str, run_id: str, payload: dict) -> None:
    last = conn.execute(text(f"SELECT seq, hash FROM {APP_SCHEMA}.audit_log ORDER BY seq DESC LIMIT 1")).first()
    seq, prev = (last[0] + 1, last[1]) if last else (1, GENESIS)
    at = datetime.now(timezone.utc).isoformat()
    entry = {"seq": seq, "event": event, "run_id": str(run_id), "payload": payload, "at": at, "prev_hash": prev}
    conn.execute(
        text(f"INSERT INTO {APP_SCHEMA}.audit_log (seq, event, run_id, payload, at, prev_hash, hash) "
             "VALUES (:seq, :event, :run_id, :payload, :at, :prev_hash, :hash)"),
        {**entry, "payload": _canonical(payload), "hash": _digest(entry)},
    )


def audit_log(limit: int = 50) -> list[dict]:
    with admin_engine().connect() as conn:
        rows = conn.execute(text(f"SELECT * FROM {APP_SCHEMA}.audit_log ORDER BY seq DESC LIMIT :n"),
                            {"n": limit}).mappings().all()
    return [{**dict(r), "payload": json.loads(r["payload"])} for r in rows]


def verify_audit() -> dict:
    """Recompute the hash chain and compare every run with the digest logged when it was saved."""
    problems: list[dict] = []
    with admin_engine().connect() as conn:
        entries = conn.execute(text(f"SELECT * FROM {APP_SCHEMA}.audit_log ORDER BY seq")).mappings().all()
        prev, logged_runs = GENESIS, set()
        for e in entries:
            payload = json.loads(e["payload"])
            expected = _digest({"seq": e["seq"], "event": e["event"], "run_id": e["run_id"], "payload": payload,
                                "at": e["at"], "prev_hash": prev})
            if e["prev_hash"] != prev or e["hash"] != expected:
                problems.append({"seq": e["seq"], "problem": "audit entry altered, removed or out of order"})
            prev = e["hash"]
            if e["event"] in ("run_recorded", "run_baselined"):
                logged_runs.add(e["run_id"])
                try:
                    current = _digest(_run_snapshot(conn, e["run_id"]))
                except Exception:
                    problems.append({"seq": e["seq"], "run_id": e["run_id"], "problem": "run deleted"})
                    continue
                if current != payload["digest"]:
                    problems.append({"seq": e["seq"], "run_id": e["run_id"],
                                     "problem": "run or its queries changed after they were recorded"})
        for (run_id,) in conn.execute(text(f"SELECT run_id FROM {APP_SCHEMA}.runs")).all():
            if str(run_id) not in logged_runs:
                problems.append({"run_id": str(run_id), "problem": "run has no audit entry (inserted outside the app?)"})
    return {"intact": not problems, "entries": len(entries), "problems": problems}
