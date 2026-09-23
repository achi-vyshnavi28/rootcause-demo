"""The semantic layer: business meaning of tables, plus deterministic builders for root-cause SQL.

The LLM decides WHAT to measure (MetricSpec). This module decides HOW to query it, so every
root-cause query has the same safe, reviewable shape.
"""

from datetime import date
from functools import lru_cache
from pathlib import Path

import yaml

from backend.agent.schemas import MetricSpec

LAYER_PATH = Path(__file__).parent / "semantic_layer.yaml"


@lru_cache
def load_layer() -> dict:
    return yaml.safe_load(LAYER_PATH.read_text(encoding="utf-8"))


def glossary_text() -> str:
    layer = load_layer()
    lines = ["Tables:"]
    for name, info in layer["tables"].items():
        key = ", ".join(info["key"]) or "none"
        lines.append(f"- {name} (key: {key}): {info['description']}")
    lines.append("Date columns on orders:")
    lines += [f"- {col}: {desc}" for col, desc in layer["date_columns"].items()]
    lines.append("Join paths from orders o:")
    for name, j in layer["joins"].items():
        lines.append(f"- {name} {j['alias']} ON {j['on']}")
    lines.append("Business definitions:")
    lines += [f"- {g}" for g in layer["glossary"]]
    return "\n".join(lines)


def _resolve_joins(names: list[str]) -> list[str]:
    """Add required parent joins (e.g. products needs order_items) in a valid order, without duplicates."""
    joins = load_layer()["joins"]
    ordered: list[str] = []

    def add(name: str) -> None:
        if name in ordered:
            return
        for parent in joins[name].get("requires", []):
            add(parent)
        ordered.append(name)

    for n in names:
        add(n)
    return ordered


def _join_sql(names: list[str], schema: str, kind: str) -> str:
    joins = load_layer()["joins"]
    return " ".join(
        f"{kind} {schema}.{n} {joins[n]['alias']} ON {joins[n]['on']}" for n in _resolve_joins(names)
    )


def _literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _check_dates(spec: MetricSpec) -> None:
    for field in ("period_a_start", "period_a_end", "period_b_start", "period_b_end"):
        date.fromisoformat(getattr(spec, field))  # raises ValueError on bad dates
    if spec.date_column not in load_layer()["date_columns"]:
        raise ValueError(f"Unknown date column {spec.date_column!r}.")


def _base_cte(spec: MetricSpec, schema: str, segment_filter: tuple[str, str] | None = None) -> str:
    _check_dates(spec)
    d = f"o.{spec.date_column}"
    a = f"({d} >= {_literal(spec.period_a_start)} AND {d} < {_literal(spec.period_a_end)})"
    b = f"({d} >= {_literal(spec.period_b_start)} AND {d} < {_literal(spec.period_b_end)})"
    where = f"({a} OR {b})"
    if spec.filter_sql:
        where += f" AND ({spec.filter_sql})"
    if segment_filter:
        dim_name, segment = segment_filter
        dim = load_layer()["dimensions"][dim_name]
        where += (
            f" AND o.order_id IN (SELECT o.order_id FROM {schema}.orders o "
            f"{_join_sql(dim['joins'], schema, 'JOIN')} WHERE CAST({dim['sql']} AS TEXT) = {_literal(segment)})"
        )
    return (
        f"base AS (SELECT o.order_id, CASE WHEN {b} THEN 'B' ELSE 'A' END AS period, "
        f"{spec.order_value_sql} AS v "
        f"FROM {schema}.orders o {_join_sql(spec.joins, schema, 'LEFT JOIN')} "
        f"WHERE {where} GROUP BY o.order_id, 2)"
    )


def total_query(spec: MetricSpec, schema: str) -> str:
    return (
        f"WITH {_base_cte(spec, schema)} "
        "SELECT period, SUM(v) AS total, COUNT(v) AS n, COUNT(*) AS orders FROM base GROUP BY period"
    )


def dimension_query(
    spec: MetricSpec, schema: str, dimension: str, segment_filter: tuple[str, str] | None = None
) -> str:
    dim = load_layer()["dimensions"][dimension]
    seg_cte = (
        f"seg AS (SELECT DISTINCT b.order_id, CAST({dim['sql']} AS TEXT) AS segment FROM base b "
        f"JOIN {schema}.orders o ON o.order_id = b.order_id {_join_sql(dim['joins'], schema, 'JOIN')})"
    )
    return (
        f"WITH {_base_cte(spec, schema, segment_filter)}, {seg_cte} "
        "SELECT COALESCE(seg.segment, '(none)') AS segment, b.period, SUM(b.v) AS total, "
        "COUNT(b.v) AS n, COUNT(*) AS orders "
        "FROM base b LEFT JOIN seg ON seg.order_id = b.order_id GROUP BY 1, 2"
    )


def duplicate_check_query(spec: MetricSpec, schema: str, table: str) -> str | None:
    """Count rows vs distinct keys per period for a table the metric reads (detects duplicated data)."""
    layer = load_layer()
    key = layer["tables"][table]["key"]
    if not key:
        return None
    _check_dates(spec)
    d = f"o.{spec.date_column}"
    b = f"({d} >= {_literal(spec.period_b_start)} AND {d} < {_literal(spec.period_b_end)})"
    a = f"({d} >= {_literal(spec.period_a_start)} AND {d} < {_literal(spec.period_a_end)})"
    if table == "orders":
        alias, join = "o", ""
    else:
        alias = layer["joins"][table]["alias"]
        join = _join_sql([table], schema, "JOIN")
    key_expr = ", ".join(f"{alias}.{k}" for k in key)
    distinct = f"COUNT(DISTINCT ({key_expr}))" if len(key) > 1 else f"COUNT(DISTINCT {key_expr})"
    return (
        f"SELECT CASE WHEN {b} THEN 'B' ELSE 'A' END AS period, COUNT(*) AS row_count, "
        f"{distinct} AS distinct_keys FROM {schema}.orders o {join} WHERE ({a} OR {b}) GROUP BY 1"
    )


def dimensions() -> list[str]:
    return list(load_layer()["dimensions"])
