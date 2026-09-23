"""Index a folder of markdown documents for a dataset.

    python -m backend.rag.ingest --dataset olist_lab --folder docs/ops_corpus/olist_lab
"""

import argparse
from pathlib import Path

from backend.rag.chunking import chunk_documents
from backend.rag.embeddings import GeminiEmbedder
from backend.rag.store import PostgresDocStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--folder", required=True, type=Path)
    args = parser.parse_args()
    chunks = chunk_documents(args.folder)
    count = PostgresDocStore(GeminiEmbedder()).ingest(args.dataset, chunks)
    print(f"Indexed {count} chunks from {len({c.doc_id for c in chunks})} documents for {args.dataset}.")


if __name__ == "__main__":
    main()
