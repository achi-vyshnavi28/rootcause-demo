"""RootCause web UI. Run with:  streamlit run frontend/app.py"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Hosted demo: expose Streamlit Cloud secrets (e.g. GEMINI_API_KEY) as environment variables for LiteLLM.
try:
    for _key, _value in st.secrets.items():
        if isinstance(_value, str):
            os.environ.setdefault(_key, _value)
except Exception:  # no secrets.toml locally: settings come from .env
    pass

from backend import config  # noqa: E402
from backend.agent.runner import run_question  # noqa: E402
from backend.db import app_store  # noqa: E402

EVAL_REPORT = ROOT / "evals" / "reports" / "latest.json"
EXAMPLES = [
    ("olist_lab", "Why did the number of orders drop in April 2018 compared to March 2018?"),
    ("olist_lab", "Why did the late delivery rate increase in June 2018 compared to May 2018?"),
    ("olist_lab", "Why did revenue jump in October 2017 compared to September 2017?"),
    ("olist", "Which 5 product categories sold the most items?"),
]

st.set_page_config(page_title="RootCause", page_icon="🔎", layout="wide")


@st.cache_resource
def _init() -> bool:
    app_store.init_app_tables()
    return True


_init()

st.title("🔎 RootCause")
st.caption("Ask why a metric changed. Every number in the answer links to the SQL that produced it.")

with st.sidebar:
    dataset = st.selectbox("Dataset", list(config.DATASETS), format_func=lambda k: config.DATASETS[k], index=1)
    st.caption(f"Model: `{config.LLM_MODEL}`")
    st.markdown("**Try an example**")
    for ds, q in EXAMPLES:
        if st.button(q, key=q, width="stretch"):
            st.session_state["question"], st.session_state["dataset_choice"] = q, ds

ask_tab, history_tab, evals_tab = st.tabs(["Ask", "History", "Evals"])


def _fmt(v, kind: str) -> str:
    return f"{v:.3f}" if kind == "per_order_average" else f"{v:,.0f}"


def show_report(result: dict) -> None:
    report, status = result["report"] or {}, result["status"]
    cols = st.columns(4)
    cols[0].metric("Status", status)
    cols[1].metric("Queries", len(result["queries"]))
    cols[2].metric("Time (s)", result["duration_s"])
    cols[3].metric("LLM cost ($)", f"{result['usage'].get('cost_usd', 0):.4f}")

    if report.get("type") == "root_cause":
        ev = report["evidence"]
        ch = ev["overall_change"]
        st.subheader(f"{ev['metric']}: {_fmt(ch['value_a'], ev['kind'])} → {_fmt(ch['value_b'], ev['kind'])}"
                     + (f" ({ch['pct_change']:+.1f}%)" if ch["pct_change"] is not None else ""))
        st.markdown(f"**Confidence:** `{report['confidence']}`")
        st.markdown(report["summary"])
        st.markdown(report["root_cause_explanation"])
        business = [c for c in ev["top_candidates"] if c["dimension"] != "data_quality"]
        if ev["data_quality_findings"]:
            st.error("Data-quality problem detected: " + ", ".join(
                f"{f['table']} duplicate rows {f['duplicate_rate_a']:.1%} → {f['duplicate_rate_b']:.1%} [{f['query_id']}]"
                for f in ev["data_quality_findings"]))
        if business:
            df = pd.DataFrame(business)
            df["label"] = df["dimension"] + " = " + df["segment"]
            fig = px.bar(df, x="share_of_change", y="label", orientation="h", hover_data=["query_id", "value_a", "value_b"],
                         title="Share of the change explained (beyond normal size)")
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=320)
            st.plotly_chart(fig, width="stretch")
        if ev.get("drill_down_inside_top_candidate"):
            st.markdown("**Inside the top segment**")
            st.dataframe(pd.DataFrame(ev["drill_down_inside_top_candidate"]), hide_index=True)
        st.markdown("**Next checks**")
        for item in report["next_checks"]:
            st.markdown(f"- {item}")
        st.markdown(f"**Suggested experiment:** {report['suggested_experiment']}")
    elif report.get("type") == "error":
        message = str(report.get("answer", ""))
        if "API key" in message or "API_KEY" in message:
            st.error("The AI model's API key is missing or invalid. Add GEMINI_API_KEY in the app's secrets / .env.")
        elif "503" in message or "high demand" in message or "RateLimit" in message:
            st.error("The free AI model is busy right now. Please try again in a minute.")
        else:
            st.error("The investigation failed. Details are below.")
        with st.expander("Technical details"):
            st.code(message[:3000])
    else:
        st.markdown(report.get("answer", ""))

    if report.get("unsupported_numbers"):
        st.warning(f"Numbers not found in the evidence (possible hallucination): {report['unsupported_numbers']}")

    with st.expander(f"Audit trail: {len(result['queries'])} queries"):
        for q in result["queries"]:
            st.markdown(f"**{q['query_id']}**: {q.get('purpose', '')}"
                        + (f"  ·  {q.get('row_count')} rows, {q.get('duration_ms')} ms" if q.get("row_count") is not None else ""))
            if q.get("error"):
                st.error(q["error"])
            st.code(q.get("sql_executed") or q.get("sql_submitted"), language="sql")
            if q.get("preview"):
                st.dataframe(pd.DataFrame(q["preview"]), hide_index=True)

    fb = st.columns([1, 1.5, 5])  # BUG-001: "Not helpful" was cut off at 1400 px
    if fb[0].button("👍 Helpful", key=f"up{result['run_id']}"):
        app_store.add_feedback(result["run_id"], 1, None)
        st.toast("Thanks!")
    if fb[1].button("👎 Not helpful", key=f"down{result['run_id']}"):
        app_store.add_feedback(result["run_id"], -1, None)
        st.toast("Thanks, noted.")


with ask_tab:
    if "dataset_choice" in st.session_state:
        dataset = st.session_state.pop("dataset_choice")
    question = st.text_input("Your question", key="question",
                             placeholder="Type a question, e.g. Why did orders drop in April 2018 compared to March 2018?")
    st.caption("Tip: pick an example in the sidebar (» at the top left on small screens). An investigation takes about 1-3 minutes.")
    clicked = st.button("Investigate", type="primary")
    if clicked and not question.strip():
        st.warning("Please type a question first. The grey text is only an example.")
    if clicked and question.strip():
        with st.spinner("Investigating: planning, querying, checking..."):
            st.session_state["last_result"] = run_question(question.strip(), dataset)
    if "last_result" in st.session_state:
        show_report(st.session_state["last_result"])

with history_tab:
    check = app_store.verify_audit()
    if check["intact"]:
        st.success(f"Audit trail intact: {check['entries']} hash-chained entries verified, and every stored answer "
                   "matches what was recorded.")
    else:
        st.error(f"Audit trail problem: {len(check['problems'])} issue(s). Stored answers may have been changed "
                 "outside the app.")
        st.dataframe(pd.DataFrame(check["problems"]), hide_index=True, width="stretch")
    history = app_store.list_runs(50)
    if history:
        st.dataframe(pd.DataFrame(history), hide_index=True, width="stretch")
    else:
        st.info("No runs yet.")

with evals_tab:
    if EVAL_REPORT.exists():
        data = json.loads(EVAL_REPORT.read_text(encoding="utf-8"))
        st.markdown(f"Run at `{data['finished_at']}` with `{data['model']}`")
        st.dataframe(pd.DataFrame([data["summary"]]).T.rename(columns={0: "value"}).astype(str), width="stretch")
        st.dataframe(pd.DataFrame(data["results"]), hide_index=True, width="stretch")
    else:
        st.info("No eval report yet. Run: python -m evals.run_evals")
