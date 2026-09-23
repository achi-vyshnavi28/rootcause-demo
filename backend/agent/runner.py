"""Entry point: run one question through the agent and (optionally) save it to the audit trail."""

import time
import uuid

from backend.agent.graph import AgentDeps, build_graph
from backend.config import DATA_SCHEMA


def default_deps() -> AgentDeps:
    from backend.agent.context import schema_context
    from backend.db.executor import run_readonly_sql
    from backend.llm import LiteLLMClient
    from backend.rag.embeddings import GeminiEmbedder
    from backend import config
    from backend.rag.store import FileDocStore, PostgresDocStore

    store = FileDocStore(GeminiEmbedder(), config.DEMO_DOCS) if config.DEMO_MODE else PostgresDocStore(GeminiEmbedder())

    def retrieve(query: str, dataset: str, start: str, end: str) -> list[dict]:
        return store.search(query, dataset, start, end, k=3)

    return AgentDeps(llm=LiteLLMClient(), run_sql=run_readonly_sql, schema_context=schema_context, retrieve=retrieve)


def run_question(question: str, dataset: str = DATA_SCHEMA, deps: AgentDeps | None = None, save: bool = True) -> dict:
    deps = deps or default_deps()
    started = time.perf_counter()
    try:
        final = build_graph(deps).invoke({"question": question, "dataset": dataset}, {"recursion_limit": 40})
        status, report, queries = final.get("status", "error"), final.get("report"), final.get("queries", [])
    except Exception as e:  # LLM outage, bad config... never crash the API
        status, report, queries = "error", {"type": "error", "answer": f"{type(e).__name__}: {e}"}, []
    result = {
        "run_id": str(uuid.uuid4()),
        "question": question,
        "dataset": dataset,
        "status": status,
        "report": report,
        "queries": queries,
        "usage": deps.llm.usage.as_dict(),
        "duration_s": round(time.perf_counter() - started, 2),
    }
    if save:
        from backend.db.app_store import save_run

        save_run(result)
    return result
