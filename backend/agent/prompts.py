"""Prompts for each LLM step. Kept short and strict: the LLM plans and explains, code does the maths."""

SYSTEM = (
    "You are RootCause, a careful senior data analyst for an e-commerce company. "
    "You only use the database described in the context. Never invent tables, columns or numbers."
)

ROUTE = """Context:
{context}

Question: {question}

Classify the question. Use "unanswerable" only if it needs data that is not in these tables
(for example website traffic, marketing spend, stock levels, employee data)."""

METRIC = """Context:
{context}

Question: {question}

Define the metric and the two periods to compare so the change can be broken down by segment.
Period B is the period the question asks about; period A is the baseline it is compared with
(if no baseline is given, use the previous period of the same length).
{previous_error}"""

SQL = """Context:
{context}

Question: {question}

Write ONE PostgreSQL SELECT query that answers the question. Use schema-qualified table names,
readable column aliases, and ORDER BY when order matters.
{previous_error}"""

ANSWER = """Question: {question}

Query {query_id} ({purpose}) returned {row_count} rows. First rows:
{rows}

Answer the question directly in 1-3 sentences using only these numbers. Cite [{query_id}]."""

REPORT = """Question: {question}

Evidence (computed by code from the queries listed; every number you write must come from here):
{evidence}

Write the report. Rules:
- Use only numbers that appear in the evidence (you may round them or show fractions as percentages).
- Cite query ids in square brackets, e.g. [Q3].
- If a data-quality finding exists, say the change is most likely a data problem, not a business change.
- related_documents are release notes / incident logs from around that time. If one clearly matches the top
  cause, mention it and cite it like [D1]; say it is supporting context, not proof. Ignore unrelated documents.
- If confidence is low or no_significant_change, say clearly that the evidence does not support a single cause."""


def previous_error_text(error: str | None) -> str:
    return f"\nYour previous attempt failed with this error, fix it:\n{error}" if error else ""
