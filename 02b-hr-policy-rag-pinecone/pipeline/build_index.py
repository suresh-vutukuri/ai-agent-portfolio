"""Ingest sample_docs/, chunk, embed, and upsert to Pinecone. Prints the final vector count."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DOCS_DIR = PROJECT_ROOT / "sample_docs"

# ingestion/ and retrieval/ are plain sibling dirs, not packages
sys.path.insert(0, str(PROJECT_ROOT / "ingestion"))
sys.path.insert(0, str(PROJECT_ROOT / "retrieval"))

from chunker import chunk_documents  # noqa: E402
from embed_store import build_vector_store  # noqa: E402
from loaders import load_documents  # noqa: E402


def main() -> None:
    """Load, chunk, embed, and upsert all sample documents to Pinecone.

    Returns:
        None. Prints progress and the final index vector count as it goes.

    Raises:
        ValueError: If the target Pinecone index (PINECONE_INDEX_NAME)
            doesn't already exist - this script never provisions one;
            create it in the Pinecone console first.
    """
    documents = load_documents(SAMPLE_DOCS_DIR)
    print(f"Loaded {len(documents)} document(s) from {SAMPLE_DOCS_DIR}")

    chunks = chunk_documents(documents)
    print(f"Produced {len(chunks)} chunk(s).")

    texts = [chunk.text for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    vector_store = build_vector_store(texts, metadatas)

    stats = vector_store.index.describe_index_stats()
    print(f"Index vector count: {stats['total_vector_count']}")


if __name__ == "__main__":
    main()
