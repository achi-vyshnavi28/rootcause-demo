"""Structured outputs the LLM must return. Pydantic validates every reply."""

from typing import Literal

from pydantic import BaseModel, Field


class Route(BaseModel):
    intent: Literal["why_change", "data_question", "unanswerable"] = Field(
        description="why_change: asks why a metric went up/down between two periods. "
        "data_question: asks for a number, list or breakdown. "
        "unanswerable: needs data that does not exist in the schema."
    )
    reason: str = Field(description="One sentence explaining the choice.")


class MetricSpec(BaseModel):
    metric_name: str = Field(description="Short human name, e.g. 'late delivery rate'.")
    kind: Literal["total", "per_order_average"] = Field(
        description="total: metric is the SUM of the per-order value (counts, revenue). "
        "per_order_average: metric is the AVERAGE of the per-order value (rates as 0/1 flags, avg review score, AOV)."
    )
    order_value_sql: str = Field(
        description="SQL aggregate computing ONE value per order, grouped by o.order_id. "
        "Wrap orders columns in MAX(...), aggregate child rows with SUM/AVG. "
        "Return NULL for orders that should not count (e.g. undelivered orders for a late rate). "
        "Examples: '1' ; 'SUM(op.payment_value)' ; 'AVG(r.review_score)' ; "
        "'MAX(CASE WHEN o.order_status = ''canceled'' THEN 1 ELSE 0 END)'"
    )
    joins: list[Literal["customers", "order_items", "order_payments", "order_reviews", "products", "sellers"]] = Field(
        default_factory=list, description="Tables order_value_sql needs besides orders."
    )
    date_column: str = Field(description="orders date column that defines the periods, usually order_purchase_timestamp.")
    filter_sql: str | None = Field(default=None, description="Optional extra condition on orders, e.g. o.order_status = 'delivered'.")
    period_a_start: str = Field(description="Baseline period start, YYYY-MM-DD (inclusive).")
    period_a_end: str = Field(description="Baseline period end, YYYY-MM-DD (exclusive).")
    period_b_start: str = Field(description="Comparison period start, YYYY-MM-DD (inclusive).")
    period_b_end: str = Field(description="Comparison period end, YYYY-MM-DD (exclusive).")


class SQLDraft(BaseModel):
    purpose: str = Field(description="What this query answers, in one line.")
    sql: str = Field(description="One PostgreSQL SELECT query.")


class Answer(BaseModel):
    answer: str = Field(description="Direct answer using only numbers from the query results, citing query ids like [Q1].")


class Narrative(BaseModel):
    summary: str = Field(description="2 sentences: what changed and the main cause, citing query ids like [Q2].")
    root_cause_explanation: str = Field(description="Why the top cause explains the change, using only the evidence numbers.")
    next_checks: list[str] = Field(description="2-3 concrete things a human should check next.")
    suggested_experiment: str = Field(description="One experiment or fix to test, with the metric to watch.")
