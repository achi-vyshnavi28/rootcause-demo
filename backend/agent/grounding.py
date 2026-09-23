"""Hallucination check: every number the LLM writes must come from the evidence.

A number in the report is "supported" if some evidence value, possibly shown as a percentage
or without its sign, rounds to it at the precision the writer used. Anything else is flagged.
"""

import re

NUMBER = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?")


def _numbers_in(value) -> list[float]:
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, str):
        return [float(m.replace(",", "")) for m in NUMBER.findall(value)]
    if isinstance(value, dict):
        return [n for v in value.values() for n in _numbers_in(v)]
    if isinstance(value, (list, tuple)):
        return [n for v in value for n in _numbers_in(v)]
    return []


def _evidence_values(evidence) -> list[float]:
    values = [float(i) for i in range(0, 11)]  # small counts like "3 sellers"
    for n in _numbers_in(evidence):
        values += [n, abs(n), n * 100, abs(n) * 100]
    return values


def _is_supported(raw: str, values: list[float]) -> bool:
    cleaned = raw.replace(",", "").lstrip("+")
    x = float(cleaned)
    decimals = len(cleaned.split(".")[1]) if "." in cleaned else 0
    tolerance = 0.5 * 10 ** (-decimals) + 1e-9
    return any(abs(abs(x) - abs(v)) <= tolerance for v in values)


def unsupported_numbers(text: str, evidence) -> list[str]:
    values = _evidence_values(evidence)
    return [raw for raw in NUMBER.findall(text) if not _is_supported(raw, values)]
