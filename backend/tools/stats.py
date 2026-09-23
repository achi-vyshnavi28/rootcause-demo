"""Root-cause maths. Deterministic, no AI: the LLM plans and explains, these functions prove.

- period_change:          how much a metric moved between period A and period B
- segment_contributions:  which segments (city, seller, category...) caused a count/sum change
- mix_rate_decomposition: for ratio metrics (conversion, late-delivery rate...), split the change into
                          mix (segment sizes shifted), rate (segments performed differently) and interaction
"""

import pandas as pd


def period_change(value_a: float, value_b: float) -> dict:
    delta = value_b - value_a
    pct = delta / value_a * 100 if value_a else None
    return {"value_a": value_a, "value_b": value_b, "delta": delta, "pct_change": pct}


def segment_contributions(
    df: pd.DataFrame, segment: str, value_a: str, value_b: str
) -> pd.DataFrame:
    """Rank segments by how much of the total change they explain.

    share_of_change is each segment's delta divided by the total delta (shares sum to 1).
    """
    out = df[[segment, value_a, value_b]].copy()
    out["delta"] = out[value_b] - out[value_a]
    total_delta = out["delta"].sum()
    out["share_of_change"] = out["delta"] / total_delta if total_delta else 0.0
    out["pct_change"] = out["delta"] / out[value_a].where(out[value_a] != 0) * 100
    return out.sort_values("delta", key=abs, ascending=False).reset_index(drop=True)


def mix_rate_decomposition(
    df: pd.DataFrame,
    segment: str,
    num_a: str,
    den_a: str,
    num_b: str,
    den_b: str,
) -> dict:
    """Decompose the change in an overall ratio (sum(num) / sum(den)) across segments.

    With w = segment weight (den / total den) and r = segment rate (num / den):
        total change = mix + rate + interaction
        mix         = sum((w_b - w_a) * r_a)
        rate        = sum(w_a * (r_b - r_a))
        interaction = sum((w_b - w_a) * (r_b - r_a))
    """
    d = df[[segment, num_a, den_a, num_b, den_b]].copy()
    w_a = d[den_a] / d[den_a].sum()
    w_b = d[den_b] / d[den_b].sum()
    r_a = (d[num_a] / d[den_a]).fillna(0.0)
    r_b = (d[num_b] / d[den_b]).fillna(0.0)

    d["weight_a"], d["weight_b"] = w_a, w_b
    d["rate_a"], d["rate_b"] = r_a, r_b
    d["mix_effect"] = (w_b - w_a) * r_a
    d["rate_effect"] = w_a * (r_b - r_a)
    d["interaction"] = (w_b - w_a) * (r_b - r_a)
    d["total_effect"] = d["mix_effect"] + d["rate_effect"] + d["interaction"]

    rate_total_a = d[num_a].sum() / d[den_a].sum()
    rate_total_b = d[num_b].sum() / d[den_b].sum()
    return {
        "rate_a": rate_total_a,
        "rate_b": rate_total_b,
        "delta": rate_total_b - rate_total_a,
        "mix_effect": d["mix_effect"].sum(),
        "rate_effect": d["rate_effect"].sum(),
        "interaction": d["interaction"].sum(),
        "segments": d.sort_values("total_effect", key=abs, ascending=False).reset_index(drop=True),
    }
