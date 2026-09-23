"""Text -> vector. Gemini embeddings in production, a deterministic hashing embedder in tests."""

import hashlib
import re
from typing import Protocol

import numpy as np

EMBEDDING_MODEL = "gemini/gemini-embedding-001"


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


class GeminiEmbedder:
    def __init__(self, model: str = EMBEDDING_MODEL, batch_size: int = 50):
        self.model, self.batch_size = model, batch_size

    def embed(self, texts: list[str]) -> np.ndarray:
        import litellm

        litellm.suppress_debug_info = True
        vectors = []
        for i in range(0, len(texts), self.batch_size):
            response = litellm.embedding(model=self.model, input=texts[i : i + self.batch_size], num_retries=3)
            vectors += [item["embedding"] for item in response.data]
        return np.asarray(vectors, dtype=np.float32)


class HashingEmbedder:
    """Bag-of-words hashed into a fixed-size vector. No network; good enough for tests."""

    def __init__(self, dims: int = 256):
        self.dims = dims

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dims), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                out[row, int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dims] += 1.0
        return out
