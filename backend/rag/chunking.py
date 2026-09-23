"""Split markdown documents (with a small front-matter header) into retrievable chunks."""

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

FRONT_MATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


@dataclass
class Chunk:
    doc_id: str
    chunk_no: int
    title: str
    doc_type: str
    doc_date: str | None
    text: str


def parse_document(path: Path) -> tuple[dict, str]:
    raw = path.read_text(encoding="utf-8")
    match = FRONT_MATTER.match(raw)
    meta = yaml.safe_load(match.group(1)) if match else {}
    body = raw[match.end():] if match else raw
    meta.setdefault("title", path.stem.replace("_", " "))
    return meta, body.strip()


def chunk_text(body: str, max_chars: int = 900) -> list[str]:
    """Greedy paragraph packing: keeps paragraphs whole, starts a new chunk when one would overflow."""
    chunks, current = [], ""
    for para in (p.strip() for p in body.split("\n\n")):
        if not para:
            continue
        if current and len(current) + len(para) + 2 > max_chars:
            chunks.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        chunks.append(current)
    return chunks


def chunk_documents(folder: Path) -> list[Chunk]:
    out = []
    for path in sorted(folder.glob("*.md")):
        meta, body = parse_document(path)
        date = str(meta["date"]) if meta.get("date") else None
        for i, text in enumerate(chunk_text(body)):
            out.append(Chunk(path.stem, i, meta["title"], meta.get("type", "document"), date, text))
    return out
