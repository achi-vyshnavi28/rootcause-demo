"""Checks every SQL query the AI writes before it is allowed to run.

Rules:
- exactly one statement, and it must be a read-only query (SELECT / WITH / UNION)
- no data-changing or admin commands anywhere inside it
- no dangerous server functions (sleep, file access, remote connections...)
- only tables from allowed schemas; unqualified tables get the default schema
- a row LIMIT is always applied (and capped)

This is the first safety layer. The read-only database user is the second.
"""

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

DEFAULT_SCHEMA = "olist"
MAX_ROWS = 1000

_FORBIDDEN_NODES = tuple(
    getattr(exp, name)
    for name in (
        "Insert", "Update", "Delete", "Merge", "Drop", "Create", "Alter", "AlterTable",
        "TruncateTable", "Copy", "Grant", "Revoke", "Command", "Set", "Transaction",
        "Commit", "Rollback", "Use", "LoadData", "Lock",
    )
    if hasattr(exp, name)
)

_FORBIDDEN_FUNCTIONS = {
    "pg_sleep", "pg_sleep_for", "pg_sleep_until",
    "pg_read_file", "pg_read_binary_file", "pg_ls_dir", "pg_stat_file",
    "lo_import", "lo_export", "dblink", "dblink_exec",
    "pg_terminate_backend", "pg_cancel_backend", "pg_reload_conf",
    "set_config", "current_setting", "query_to_xml",
}


class SQLValidationError(ValueError):
    """Raised when a query is not safe to run. The message is fed back to the LLM."""


def _function_name(node: exp.Func) -> str:
    return (node.name if isinstance(node, exp.Anonymous) else node.sql_name()).lower()


def validate_sql(
    sql: str,
    allowed_schemas: frozenset[str] = frozenset({DEFAULT_SCHEMA}),
    default_schema: str = DEFAULT_SCHEMA,
    max_rows: int = MAX_ROWS,
) -> str:
    """Return a safe, normalized version of `sql`, or raise SQLValidationError."""
    try:
        statements = [s for s in sqlglot.parse(sql, read="postgres") if s is not None]
    except ParseError as e:
        raise SQLValidationError(f"SQL could not be parsed: {e}") from e

    if len(statements) != 1:
        raise SQLValidationError(f"Exactly one statement is allowed, got {len(statements)}.")
    tree = statements[0]

    if not isinstance(tree, exp.Query):
        raise SQLValidationError(f"Only read queries are allowed, got {tree.key.upper()}.")

    for node in tree.walk():
        if isinstance(node, _FORBIDDEN_NODES):
            raise SQLValidationError(f"Forbidden operation: {node.key.upper()}.")
        if isinstance(node, exp.Func) and _function_name(node) in _FORBIDDEN_FUNCTIONS:
            raise SQLValidationError(f"Forbidden function: {_function_name(node)}.")
        if isinstance(node, exp.Select) and (node.args.get("into") or node.args.get("locks")):
            raise SQLValidationError("SELECT INTO and row locks (FOR UPDATE) are not allowed.")

    cte_names = {cte.alias_or_name for cte in tree.find_all(exp.CTE)}
    for table in tree.find_all(exp.Table):
        if not table.name:
            continue  # table functions such as generate_series
        if table.db:
            if table.db not in allowed_schemas:
                raise SQLValidationError(f"Schema '{table.db}' is not allowed.")
        elif table.name not in cte_names:
            table.set("db", exp.to_identifier(default_schema))

    limit = tree.args.get("limit")
    limit_value = limit.expression if limit is not None else None
    within_cap = (
        isinstance(limit_value, exp.Literal)
        and limit_value.is_int
        and int(limit_value.this) <= max_rows
    )
    if not within_cap:
        tree = tree.limit(max_rows)

    return tree.sql(dialect="postgres")
