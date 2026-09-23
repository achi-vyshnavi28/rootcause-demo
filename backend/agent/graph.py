"""The investigation workflow, as a LangGraph state machine.

    load_context -> route -+-> define_metric -> measure -> drill_down -> check_quality -> find_context -> write_report
                           |        ^______________| (retry on SQL error)
                           +-> write_sql -> run_sql -> answer
                           |      ^____________| (retry on SQL error)
                           +-> refuse
"""

import json
from dataclasses import dataclass
from typing import Callable, TypedDict

import pandas as pd
from langgraph.graph import END, StateGraph

from backend import config
from backend.agent import analysis, prompts, semantic
from backend.agent.grounding import unsupported_numbers
from backend.agent.schemas import Answer, MetricSpec, Narrative, Route, SQLDraft
from backend.db.executor import QueryResult
from backend.guardrails.sql_validator import SQLValidationError, validate_sql
from backend.llm import LLM

ANALYSIS_MAX_ROWS = 20_000  # dimension queries return one row per segment and period (thousands of sellers)


@dataclass
class AgentDeps:
    llm: LLM
    run_sql: Callable[[str], QueryResult]
    schema_context: Callable[[str], str]
    max_retries: int = config.LLM_MAX_SQL_RETRIES
    # (query, dataset, period_start, period_end) -> document chunks; None disables document search
    retrieve: Callable[[str, str, str, str], list[dict]] | None = None


class AgentState(TypedDict, total=False):
    question: str
    dataset: str
    context: str
    intent: str
    route_reason: str
    metric: dict
    attempts: int
    error: str | None
    queries: list[dict]
    change: dict
    change_query_id: str
    dimension_query_ids: dict
    candidates: list[dict]
    drilldown: list[dict]
    quality: list[dict]
    documents: list[dict]
    report: dict
    status: str


class QueryFailed(Exception):
    pass


def _execute(state: AgentState, deps: AgentDeps, purpose: str, sql: str, max_rows: int = 1000) -> tuple[pd.DataFrame, str]:
    """Validate, run and log one query in the audit trail. Raises QueryFailed with a message for the LLM."""
    dataset = state["dataset"]
    query_id = f"Q{len(state['queries']) + 1}"
    entry = {"query_id": query_id, "purpose": purpose, "sql_submitted": sql}
    state["queries"].append(entry)
    try:
        safe_sql = validate_sql(sql, allowed_schemas=frozenset({dataset}), default_schema=dataset, max_rows=max_rows)
    except SQLValidationError as e:
        entry["error"] = f"Rejected by validator: {e}"
        raise QueryFailed(entry["error"]) from e
    entry["sql_executed"] = safe_sql
    try:
        result = deps.run_sql(safe_sql)
    except Exception as e:
        entry["error"] = f"Database error: {str(e).splitlines()[0][:500]}"
        raise QueryFailed(entry["error"]) from e
    entry["row_count"] = len(result.df)
    entry["duration_ms"] = result.duration_ms
    entry["preview"] = json.loads(result.df.head(10).to_json(orient="records", date_format="iso"))
    return result.df, query_id


def build_graph(deps: AgentDeps):
    llm = deps.llm

    def load_context(state: AgentState) -> AgentState:
        return {"context": deps.schema_context(state["dataset"]), "queries": [], "attempts": 0, "error": None}

    def route(state: AgentState) -> AgentState:
        r = llm.complete_json(prompts.SYSTEM, prompts.ROUTE.format(**state), Route)
        return {"intent": r.intent, "route_reason": r.reason}

    # ---------- why-did-it-change path ----------
    def define_metric(state: AgentState) -> AgentState:
        prompt = prompts.METRIC.format(**state, previous_error=prompts.previous_error_text(state.get("error")))
        spec = llm.complete_json(prompts.SYSTEM, prompt, MetricSpec)
        return {"metric": spec.model_dump(), "attempts": state["attempts"] + 1, "error": None}

    def measure(state: AgentState) -> AgentState:
        spec = MetricSpec(**state["metric"])
        schema = state["dataset"]
        try:
            total_df, total_qid = _execute(state, deps, f"Total {spec.metric_name}, period A vs B", semantic.total_query(spec, schema))
            change = analysis.total_change(total_df, spec.kind)
            rankings, qids = {}, {}
            for dim in semantic.dimensions():
                df, qid = _execute(state, deps, f"{spec.metric_name} by {dim}", semantic.dimension_query(spec, schema, dim), ANALYSIS_MAX_ROWS)
                rankings[dim] = analysis.rank_segments(df, spec.kind, change["delta"])
                qids[dim] = qid
        except (QueryFailed, ValueError) as e:
            return {"error": str(e), "queries": state["queries"]}
        candidates = analysis.top_candidates(rankings)
        for c in candidates:
            c["query_id"] = qids[c["dimension"]]
        return {
            "queries": state["queries"],
            "change": change,
            "change_query_id": total_qid,
            "dimension_query_ids": qids,
            "candidates": candidates,
            "error": None,
        }

    def after_measure(state: AgentState) -> str:
        if state.get("error"):
            return "define_metric" if state["attempts"] < deps.max_retries else "fail"
        return "drill_down"

    def drill_down(state: AgentState) -> AgentState:
        """Look inside the top segment: e.g. within state SP, which seller or category drives it?"""
        if not state["candidates"]:
            return {"drilldown": []}
        spec = MetricSpec(**state["metric"])
        top = state["candidates"][0]
        results = []
        for dim in semantic.dimensions():
            if dim == top["dimension"]:
                continue
            try:
                df, qid = _execute(
                    state, deps, f"Inside {top['dimension']}={top['segment']}: {spec.metric_name} by {dim}",
                    semantic.dimension_query(spec, state["dataset"], dim, (top["dimension"], top["segment"])),
                    ANALYSIS_MAX_ROWS,
                )
            except QueryFailed:
                continue  # drill-down is optional evidence; the main analysis already succeeded
            inner_change = analysis.total_change(
                df.groupby("period", as_index=False)[["total", "n"]].sum(), spec.kind
            )
            ranked = analysis.rank_segments(df, spec.kind, inner_change["delta"])
            if not ranked.empty and ranked.loc[0, "segment"] != "(none)":
                row = ranked.loc[0]
                results.append({
                    "dimension": dim, "segment": str(row["segment"]), "value_a": float(row["value_a"]),
                    "value_b": float(row["value_b"]), "share_of_change_within": float(row["share_of_change"]),
                    "query_id": qid,
                })
        results.sort(key=lambda r: abs(r["share_of_change_within"]), reverse=True)
        return {"drilldown": results[:3], "queries": state["queries"]}

    def check_quality(state: AgentState) -> AgentState:
        spec = MetricSpec(**state["metric"])
        tables = ["orders", *semantic._resolve_joins(spec.joins)]
        checks, qids = {}, {}
        for table in tables:
            sql = semantic.duplicate_check_query(spec, state["dataset"], table)
            if not sql:
                continue
            try:
                checks[table], qids[table] = _execute(state, deps, f"Duplicate-row check on {table}", sql)
            except QueryFailed:
                continue
        findings = analysis.duplicate_findings(checks)
        for f in findings:
            f["query_id"] = qids[f["table"]]
        candidates = state["candidates"]
        if findings:  # a data bug outranks every business explanation
            dq = [{"dimension": "data_quality", "segment": f["table"], "share_of_change": None,
                   "duplicate_rate_b": f["duplicate_rate_b"], "query_id": f["query_id"]} for f in findings]
            candidates = dq + candidates
        return {"quality": findings, "candidates": candidates, "queries": state["queries"]}

    def find_context(state: AgentState) -> AgentState:
        """Search release notes / incident logs from around period B for events matching the top suspect."""
        if deps.retrieve is None:
            return {"documents": []}
        metric = state["metric"]
        top = state["candidates"][0] if state["candidates"] else {}
        query = f"{metric['metric_name']} {top.get('dimension', '')} {top.get('segment', '')} {state['question']}"
        try:
            hits = deps.retrieve(query, state["dataset"], metric["period_b_start"], metric["period_b_end"])
        except Exception:
            hits = []  # documents are supporting context; never fail the investigation over them
        documents = [
            {"doc_ref": f"D{i}", "doc_id": h["doc_id"], "title": h["title"], "type": h["doc_type"],
             "date": h["doc_date"], "excerpt": h["text"][:500]}
            for i, h in enumerate(hits, start=1)
        ]
        return {"documents": documents}

    def write_report(state: AgentState) -> AgentState:
        conf = analysis.confidence([c for c in state["candidates"] if c["dimension"] != "data_quality"], state["change"], state["quality"])
        evidence = {
            "metric": state["metric"]["metric_name"],
            "kind": state["metric"]["kind"],
            "period_a": [state["metric"]["period_a_start"], state["metric"]["period_a_end"]],
            "period_b": [state["metric"]["period_b_start"], state["metric"]["period_b_end"]],
            "overall_change": {**state["change"], "query_id": state["change_query_id"]},
            "top_candidates": state["candidates"][:5],
            "drill_down_inside_top_candidate": state.get("drilldown", []),
            "data_quality_findings": state["quality"],
            "related_documents": state.get("documents", []),
            "confidence": conf,
        }
        evidence_json = json.dumps(evidence, default=str, indent=1)
        narrative = llm.complete_json(prompts.SYSTEM, prompts.REPORT.format(question=state["question"], evidence=evidence_json), Narrative)
        text_fields = " ".join([narrative.summary, narrative.root_cause_explanation, *narrative.next_checks, narrative.suggested_experiment])
        status = {"low": "insufficient_evidence", "no_significant_change": "no_significant_change"}.get(conf, "ok")
        report = {
            "type": "root_cause",
            **narrative.model_dump(),
            "confidence": conf,
            "evidence": evidence,
            "unsupported_numbers": unsupported_numbers(text_fields, evidence),
        }
        return {"report": report, "status": status}

    # ---------- plain data-question path ----------
    def write_sql(state: AgentState) -> AgentState:
        prompt = prompts.SQL.format(**state, previous_error=prompts.previous_error_text(state.get("error")))
        draft = llm.complete_json(prompts.SYSTEM, prompt, SQLDraft)
        return {"metric": draft.model_dump(), "attempts": state["attempts"] + 1, "error": None}

    def run_sql(state: AgentState) -> AgentState:
        try:
            _execute(state, deps, state["metric"]["purpose"], state["metric"]["sql"])
        except QueryFailed as e:
            return {"error": str(e), "queries": state["queries"]}
        return {"error": None, "queries": state["queries"]}

    def after_run_sql(state: AgentState) -> str:
        if state.get("error"):
            return "write_sql" if state["attempts"] < deps.max_retries else "fail"
        return "answer"

    def answer(state: AgentState) -> AgentState:
        q = state["queries"][-1]
        rows = json.dumps(q["preview"], default=str)
        prompt = prompts.ANSWER.format(question=state["question"], query_id=q["query_id"], purpose=q["purpose"], row_count=q["row_count"], rows=rows)
        result = llm.complete_json(prompts.SYSTEM, prompt, Answer)
        return {
            "report": {
                "type": "answer",
                "answer": result.answer,
                "query_id": q["query_id"],
                "unsupported_numbers": unsupported_numbers(result.answer, [q["preview"], q["row_count"], state["question"]]),
            },
            "status": "ok",
        }

    def refuse(state: AgentState) -> AgentState:
        return {"report": {"type": "refusal", "answer": f"I can't answer this from the available data: {state['route_reason']}"}, "status": "unanswerable"}

    def fail(state: AgentState) -> AgentState:
        return {"report": {"type": "error", "answer": f"The investigation failed after {state['attempts']} attempts. Last error: {state.get('error')}"}, "status": "error"}

    g = StateGraph(AgentState)
    for name, fn in [
        ("load_context", load_context), ("route", route), ("define_metric", define_metric), ("measure", measure),
        ("drill_down", drill_down), ("check_quality", check_quality), ("find_context", find_context), ("write_report", write_report),
        ("write_sql", write_sql), ("run_sql", run_sql), ("answer", answer), ("refuse", refuse), ("fail", fail),
    ]:
        g.add_node(name, fn)
    g.set_entry_point("load_context")
    g.add_edge("load_context", "route")
    g.add_conditional_edges("route", lambda s: {"why_change": "define_metric", "data_question": "write_sql"}.get(s["intent"], "refuse"))
    g.add_edge("define_metric", "measure")
    g.add_conditional_edges("measure", after_measure)
    g.add_edge("drill_down", "check_quality")
    g.add_edge("check_quality", "find_context")
    g.add_edge("find_context", "write_report")
    g.add_edge("write_report", END)
    g.add_edge("write_sql", "run_sql")
    g.add_conditional_edges("run_sql", after_run_sql)
    g.add_edge("answer", END)
    g.add_edge("refuse", END)
    g.add_edge("fail", END)
    return g.compile()
