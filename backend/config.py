"""Reads settings from the .env file and builds database connections."""

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import URL, create_engine
from sqlalchemy.engine import Engine

# Load variables from .env into the environment (does nothing if .env is missing).
load_dotenv()

# All Olist tables live in their own schema (a named "folder" inside the database).
DATA_SCHEMA = "olist"
# Copy of the Olist data with planted anomalies (known root causes) used by the evals.
LAB_SCHEMA = "olist_lab"
# RootCause's own tables: runs, audit trail, feedback.
APP_SCHEMA = "rootcause_app"

DATASETS = {DATA_SCHEMA: "Olist e-commerce (original)", LAB_SCHEMA: "Olist lab (planted anomalies)"}

# LLM settings. Model names use LiteLLM's "provider/model" format, so switching provider is config only.
LLM_MODEL = os.getenv("LLM_MODEL", "gemini/gemini-3.6-flash")
LLM_FALLBACK_MODELS = [m.strip() for m in os.getenv("LLM_FALLBACK_MODELS", "gemini/gemini-flash-lite-latest").split(",") if m.strip()]
# Gemini 3+ "thinks" before answering; low keeps quality while cutting latency. Empty = provider default.
LLM_REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "low") or None
LLM_MAX_SQL_RETRIES = int(os.getenv("LLM_MAX_SQL_RETRIES", "3"))


# Demo mode (the hosted Hugging Face Space): business data comes from a bundled DuckDB file opened
# read-only, and RootCause's own tables live in SQLite. No database server needed.
DEMO_DIR = Path(os.getenv("ROOTCAUSE_DEMO_DIR", Path(__file__).resolve().parents[1] / "data" / "demo"))
DEMO_DB = DEMO_DIR / "rootcause_demo.duckdb"
# On by request (ROOTCAUSE_DEMO=1), or automatically when there is no PostgreSQL password but the demo file exists.
DEMO_MODE = os.getenv("ROOTCAUSE_DEMO", "") == "1" or (os.getenv("ROOTCAUSE_DEMO", "") != "0" and not os.getenv("PGPASSWORD") and DEMO_DB.exists())
DEMO_DOCS = DEMO_DIR / "doc_chunks.npz"
APP_SQLITE = Path(os.getenv("ROOTCAUSE_APP_DB", DEMO_DIR / "rootcause_app.sqlite3"))


def _required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing setting {name}. Add it to your .env file (see .env.example).")
    return value


def _build_url(user: str, password: str) -> URL:
    # URL.create safely handles special characters (like @ or #) in passwords.
    return URL.create(
        drivername="postgresql+psycopg",
        username=user,
        password=password,
        host=os.getenv("PGHOST", "127.0.0.1"),
        port=int(os.getenv("PGPORT", "5432")),
        database=os.getenv("PGDATABASE", "rootcause"),
    )


@lru_cache
def admin_engine() -> Engine:
    """Full-access connection. Only for setup scripts and RootCause's own app tables.

    In demo mode this is a local SQLite file; the app schema is attached under the same name
    so table references like rootcause_app.runs work unchanged.
    """
    if DEMO_MODE:
        from sqlalchemy import event

        APP_SQLITE.parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine("sqlite://")
        path = APP_SQLITE.as_posix()

        @event.listens_for(engine, "connect")
        def _attach(dbapi_conn, _record):
            dbapi_conn.execute(f"ATTACH DATABASE '{path}' AS {APP_SCHEMA}")

        return engine
    return create_engine(_build_url(_required("PGUSER"), _required("PGPASSWORD")), pool_pre_ping=True)


def demo_connection():
    """Read-only DuckDB connection to the bundled demo data (writes are impossible)."""
    import duckdb

    return duckdb.connect(str(DEMO_DB), read_only=True)


@lru_cache
def readonly_engine() -> Engine:
    """Read-only connection. This is the only one the AI agent uses to query business data."""
    return create_engine(
        _build_url(_required("READONLY_DB_USER"), _required("READONLY_DB_PASSWORD")),
        pool_pre_ping=True,
    )
