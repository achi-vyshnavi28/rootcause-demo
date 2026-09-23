"""Builds the text the LLM reads to understand the database: tables, columns, date range, glossary."""

from functools import lru_cache

from sqlalchemy import text

from backend.agent.semantic import dimensions, glossary_text
from backend import config
from backend.config import readonly_engine


COLUMNS_SQL = ("SELECT table_name, column_name, data_type FROM information_schema.columns "
               "WHERE table_schema = {p} ORDER BY table_name, ordinal_position")


@lru_cache
def schema_context(dataset: str) -> str:
    if config.DEMO_MODE:
        with config.demo_connection() as conn:
            columns = conn.execute(COLUMNS_SQL.format(p="?"), [dataset]).fetchall()
            date_range = conn.execute(
                f"SELECT MIN(order_purchase_timestamp), MAX(order_purchase_timestamp) FROM {dataset}.orders").fetchone()
        return format_context(dataset, columns, date_range)
    with readonly_engine().connect() as conn:
        columns = conn.execute(
            text(COLUMNS_SQL.format(p=":s")),
            {"s": dataset},
        ).all()
        date_range = conn.execute(
            text(f"SELECT MIN(order_purchase_timestamp), MAX(order_purchase_timestamp) FROM {dataset}.orders")
        ).one()
    return format_context(dataset, columns, date_range)


def format_context(dataset: str, columns, date_range) -> str:
    tables: dict[str, list[str]] = {}
    for table, column, dtype in columns:
        tables.setdefault(table, []).append(f"{column} ({dtype})")
    lines = [f"PostgreSQL schema: {dataset}. Always write {dataset}.<table>."]
    lines += [f"- {dataset}.{t}: {', '.join(cols)}" for t, cols in tables.items()]
    lines.append(f"Orders are placed between {date_range[0]} and {date_range[1]}. "
                 f"'Last month' or 'recently' means relative to {date_range[1]}.")
    lines.append(glossary_text())
    lines.append("Root-cause dimensions available: " + ", ".join(dimensions()))
    return "\n".join(lines)
