"""Turns query results into ranked root-cause candidates. Pure pandas, fully testable.

Key idea: a big segment explaining a big share of a change is NOT a root cause by itself
(Sao Paulo is ~40% of orders, so it "explains" ~40% of any change). We rank segments by their
ABNORMAL change: how far they moved beyond what their normal size predicts.
"""

import pandas as pd

from backend.tools.stats import mix_rate_decomposition


def _pivot(df: pd.DataFrame) -> pd.DataFrame:
    wide = df.pivot_table(index="segment", columns="period", values=["total", "n"], aggfunc="sum", fill_value=0)
    out = pd.DataFrame(index=wide.index)
    for metric in ("total", "n"):
        for period in ("A", "B"):
            col = (metric, period)
            out[f"{metric}_{period.lower()}"] = wide[col] if col in wide.columns else 0
    return out.reset_index().astype({"total_a": float, "total_b": float, "n_a": float, "n_b": float})


def total_change(df: pd.DataFrame, kind: str) -> dict:
    """Overall metric value in periods A and B from the total query."""
    by_period = df.set_index("period")
    values = {}
    for p in ("A", "B"):
        total = float(by_period.at[p, "total"]) if p in by_period.index and pd.notna(by_period.at[p, "total"]) else 0.0
        n = float(by_period.at[p, "n"]) if p in by_period.index else 0.0
        values[p] = (total / n if n else 0.0) if kind == "per_order_average" else total
    delta = values["B"] - values["A"]
    pct = delta / values["A"] * 100 if values["A"] else None
    return {"value_a": values["A"], "value_b": values["B"], "delta": delta, "pct_change": pct}


def rank_segments(df: pd.DataFrame, kind: str, overall_delta: float) -> pd.DataFrame:
    """One row per segment with its abnormal contribution, sorted most important first."""
    d = _pivot(df)
    if d.empty:
        return d
    if kind == "total":
        d["value_a"], d["value_b"] = d["total_a"], d["total_b"]
        d["delta"] = d["value_b"] - d["value_a"]
        weight_a = d["value_a"] / d["value_a"].sum() if d["value_a"].sum() else 0.0
        d["expected_delta"] = weight_a * d["delta"].sum()
        d["abnormal"] = d["delta"] - d["expected_delta"]
    else:
        m = mix_rate_decomposition(d, "segment", "total_a", "n_a", "total_b", "n_b")
        seg = m["segments"].set_index("segment")
        d = d.set_index("segment").join(seg[["rate_a", "rate_b", "weight_a", "weight_b", "mix_effect", "rate_effect", "interaction"]]).reset_index()
        d["value_a"], d["value_b"] = d["rate_a"], d["rate_b"]
        d["delta"] = d["value_b"] - d["value_a"]
        d["abnormal"] = d["rate_effect"] + d["interaction"]
    d["share_of_change"] = d["abnormal"] / overall_delta if overall_delta else 0.0
    return d.sort_values("abnormal", key=abs, ascending=False).reset_index(drop=True)


def top_candidates(rankings: dict[str, pd.DataFrame], per_dimension: int = 3) -> list[dict]:
    """Merge every dimension's top segments into one list ranked by share of the change explained."""
    candidates = []
    for dimension, ranked in rankings.items():
        for _, row in ranked.head(per_dimension).iterrows():
            if row["segment"] == "(none)":
                continue
            candidates.append(
                {
                    "dimension": dimension,
                    "segment": str(row["segment"]),
                    "value_a": round(float(row["value_a"]), 4),
                    "value_b": round(float(row["value_b"]), 4),
                    "change": round(float(row["delta"]), 4),
                    "abnormal_change": round(float(row["abnormal"]), 4),
                    "share_of_change": round(float(row["share_of_change"]), 4),
                }
            )
    candidates.sort(key=lambda c: abs(c["share_of_change"]), reverse=True)
    return candidates


def duplicate_findings(checks: dict[str, pd.DataFrame], min_increase: float = 0.005) -> list[dict]:
    """Flag tables whose duplicate-row rate jumped in period B (a data bug, not a business cause)."""
    findings = []
    for table, df in checks.items():
        rates = {}
        for _, row in df.iterrows():
            rows, distinct = float(row["row_count"]), float(row["distinct_keys"])
            rates[row["period"]] = (rows - distinct) / rows if rows else 0.0
        increase = rates.get("B", 0.0) - rates.get("A", 0.0)
        if increase > min_increase:
            findings.append(
                {
                    "table": table,
                    "duplicate_rate_a": rates.get("A", 0.0),
                    "duplicate_rate_b": rates.get("B", 0.0),
                    "increase": increase,
                }
            )
    return findings


def confidence(candidates: list[dict], change: dict, quality: list[dict]) -> str:
    if quality:
        return "high"
    if change["pct_change"] is not None and abs(change["pct_change"]) < 2:
        return "no_significant_change"
    if not candidates:
        return "low"
    top = abs(candidates[0]["share_of_change"])
    return "high" if top >= 0.5 else "medium" if top >= 0.25 else "low"
