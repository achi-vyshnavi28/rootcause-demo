"""Hybrid retrieval: keyword (BM25) + meaning (embeddings), merged with reciprocal rank fusion.

BM25 catches exact terms (a seller id, "boleto"); embeddings catch paraphrases
("payments were counted twice" ~ "duplicate payment rows"). RRF merges both rankings
without having to calibrate their scores against each other.
"""

import re

import numpy as np
from rank_bm25 import BM25Okapi

RRF_K = 60


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def cosine_scores(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1) * (np.linalg.norm(query) or 1.0)
    return (matrix @ query) / np.where(norms == 0, 1.0, norms)


def hybrid_rank(texts: list[str], embeddings: np.ndarray, query: str, query_embedding: np.ndarray, k: int = 3) -> list[tuple[int, float]]:
    """Return (index, fused_score) for the top-k texts."""
    if not texts:
        return []
    bm25 = BM25Okapi([_tokens(t) for t in texts]).get_scores(_tokens(query))
    dense = cosine_scores(embeddings, query_embedding)
    fused = np.zeros(len(texts))
    for scores in (bm25, dense):
        for rank, idx in enumerate(np.argsort(-scores)):
            fused[idx] += 1.0 / (RRF_K + rank + 1)
    top = np.argsort(-fused)[:k]
    return [(int(i), float(fused[i])) for i in top]
