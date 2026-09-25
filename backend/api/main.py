"""REST API. Run locally with:  uvicorn backend.api.main:app --reload"""

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from backend import config
from backend.agent.runner import run_question
from backend.db import app_store

EVAL_REPORT = Path(__file__).resolve().parents[2] / "evals" / "reports" / "latest.json"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    app_store.init_app_tables()
    yield


app = FastAPI(title="RootCause", description="AI analyst that finds why a metric changed.", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    dataset: str = config.DATA_SCHEMA


class FeedbackRequest(BaseModel):
    rating: int = Field(ge=-1, le=1, description="1 = helpful, -1 = not helpful")
    comment: str | None = Field(default=None, max_length=1000)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": config.LLM_MODEL}


@app.get("/datasets")
def datasets() -> dict:
    return config.DATASETS


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    if req.dataset not in config.DATASETS:
        raise HTTPException(400, f"Unknown dataset. Choose one of {list(config.DATASETS)}")
    return run_question(req.question, req.dataset)


@app.get("/runs")
def runs(limit: int = 20) -> list[dict]:
    return app_store.list_runs(limit)


@app.get("/runs/{run_id}")
def run(run_id: str) -> dict:
    found = app_store.get_run(run_id)
    if not found:
        raise HTTPException(404, "Run not found")
    return found


@app.get("/runs/{run_id}/audit")
def audit(run_id: str) -> list[dict]:
    return app_store.get_queries(run_id)


@app.post("/runs/{run_id}/feedback")
def feedback(run_id: str, req: FeedbackRequest) -> dict:
    if not app_store.get_run(run_id):
        raise HTTPException(404, "Run not found")
    app_store.add_feedback(run_id, req.rating, req.comment)
    return {"saved": True}


@app.get("/audit/log")
def audit_log(limit: int = 50) -> list[dict]:
    """Tamper-evident log: one hash-chained entry per saved run and per feedback, newest first."""
    return app_store.audit_log(limit)


@app.get("/audit/verify")
def audit_verify() -> dict:
    """Recompute the hash chain and compare every run with what was logged (21 CFR Part 11 §11.10(e))."""
    return app_store.verify_audit()


@app.get("/anomalies")
def anomalies(dataset: str = config.LAB_SCHEMA, threshold: float = 6.0) -> list[dict]:
    """Unusual days in key metrics (STL + robust z-score). Each one is a good question to ask RootCause."""
    if dataset not in config.DATASETS:
        raise HTTPException(400, f"Unknown dataset. Choose one of {list(config.DATASETS)}")
    from ml.anomaly import scan

    return scan(dataset, threshold)


@app.get("/evals/latest")
def latest_evals() -> dict:
    if not EVAL_REPORT.exists():
        raise HTTPException(404, "No eval report yet. Run: python -m evals.run_evals")
    return json.loads(EVAL_REPORT.read_text(encoding="utf-8"))
