"""Document chunks + embeddings in PostgreSQL, searched with hybrid retrieval.

Locally (Windows PostgreSQL without pgvector) vectors are stored as REAL[] and ranked in numpy,
which is fine for thousands of chunks. On a host with pgvector (e.g. Neon), the same table
can move to a VECTOR column with an HNSW index; the search interface stays the same.
"""

from dataclasses import asdict
from datetime import date, timedelta

import numpy as np
from sqlalchemy import text

from backend.config import APP_SCHEMA, admin_engine
from backend.rag.chunking import Chunk
from backend.rag.embeddings import Embedder
from backend.rag.search import hybrid_rank

DDL = f"""
CREATE TABLE IF NOT EXISTS {APP_SCHEMA}.doc_chunks (
    dataset   TEXT NOT NULL,
    doc_id    TEXT NOT NULL,
    chunk_no  INTEGER NOT NULL,
    title     TEXT,
    doc_type  TEXT,
    doc_date  DATE,
    text      TEXT NOT NULL,
    embedding REAL[] NOT NULL,
    PRIMARY KEY (dataset, doc_id, chunk_no)
)
"""


def _within(doc_date, start: str | None, end: str | None, margin_days: int) -> bool:
    if doc_date is None or (start is None and end is None):
        return True
    d = doc_date if isinstance(doc_date, date) else date.fromisoformat(str(doc_date))
    if start and d < date.fromisoformat(start) - timedelta(days=margin_days):
        return False
    if end and d > date.fromisoformat(end) + timedelta(days=margin_days):
        return False
    return True


class InMemoryDocStore:
    """Same interface as PostgresDocStore; used in tests."""

    def __init__(self, embedder: Embedder):
        self.embedder, self.rows = embedder, []

    def ingest(self, dataset: str, chunks: list[Chunk]) -> int:
        vectors = self.embedder.embed([c.text for c in chunks])
        self.rows = [r for r in self.rows if r["dataset"] != dataset]
        self.rows += [{**asdict(c), "dataset": dataset, "embedding": v} for c, v in zip(chunks, vectors)]
        return len(chunks)

    def _candidates(self, dataset, start, end, margin_days):
        return [r for r in self.rows if r["dataset"] == dataset and _within(r["doc_date"], start, end, margin_days)]

    def search(self, query: str, dataset: str, start: str | None = None, end: str | None = None,
               k: int = 3, margin_days: int = 20) -> list[dict]:
        rows = self._candidates(dataset, start, end, margin_days)
        if not rows:
            return []
        matrix = np.vstack([r["embedding"] for r in rows])
        hits = hybrid_rank([r["text"] for r in rows], matrix, query, self.embedder.embed([query])[0], k)
        return [{key: rows[i][key] for key in ("doc_id", "chunk_no", "title", "doc_type", "doc_date", "text")} | {"score": round(s, 4)}
                for i, s in hits]


class PostgresDocStore(InMemoryDocStore):
    def ingest(self, dataset: str, chunks: list[Chunk]) -> int:
        vectors = self.embedder.embed([c.text for c in chunks])
        with admin_engine().begin() as conn:
            conn.execute(text(DDL))
            conn.execute(text(f"DELETE FROM {APP_SCHEMA}.doc_chunks WHERE dataset = :d"), {"d": dataset})
            for c, v in zip(chunks, vectors):
                conn.execute(
                    text(f"INSERT INTO {APP_SCHEMA}.doc_chunks VALUES (:dataset, :doc_id, :chunk_no, :title, :doc_type, :doc_date, :text, :embedding)"),
                    {**asdict(c), "dataset": dataset, "embedding": [float(x) for x in v]},
                )
        return len(chunks)

    def _candidates(self, dataset, start, end, margin_days):
        with admin_engine().connect() as conn:
            conn.execute(text(DDL))
            rows = conn.execute(
                text(f"SELECT doc_id, chunk_no, title, doc_type, doc_date, text, embedding FROM {APP_SCHEMA}.doc_chunks WHERE dataset = :d"),
                {"d": dataset},
            ).mappings().all()
        return [
            {**r, "doc_date": r["doc_date"].isoformat() if r["doc_date"] else None, "embedding": np.asarray(r["embedding"], dtype=np.float32)}
            for r in rows if _within(r["doc_date"], start, end, margin_days)
        ]


class FileDocStore(InMemoryDocStore):
    """Demo mode: chunks and their embeddings exported to one .npz file (see deploy/build_demo.py)."""

    def __init__(self, embedder: Embedder, path):
        super().__init__(embedder)
        import json

        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data["meta"]))
        self.rows = [{**m, "embedding": v} for m, v in zip(meta, data["embeddings"])]


def export_chunks(path) -> int:
    """Write every indexed chunk (all datasets) from PostgreSQL to an .npz file for demo mode."""
    import json

    with admin_engine().connect() as conn:
        records = conn.execute(text(f"SELECT dataset, doc_id, chunk_no, title, doc_type, doc_date, text, embedding "
                                    f"FROM {APP_SCHEMA}.doc_chunks ORDER BY dataset, doc_id, chunk_no")).mappings().all()
    meta = [{k: (r[k].isoformat() if k == "doc_date" and r[k] else r[k]) for k in
             ("dataset", "doc_id", "chunk_no", "title", "doc_type", "doc_date", "text")} for r in records]
    np.savez_compressed(path, meta=json.dumps(meta), embeddings=np.asarray([r["embedding"] for r in records], dtype=np.float32))
    return len(records)
