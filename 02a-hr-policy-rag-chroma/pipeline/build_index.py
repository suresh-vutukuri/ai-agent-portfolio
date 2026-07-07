"""Ingest sample_docs/, chunk, embed, and persist to Chroma. Prints the final count."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DOCS_DIR = PROJECT_ROOT / "sample_docs"

# ingestion/ and retrieval/ are plain sibling dirs, not packages
sys.path.insert(0, str(PROJECT_ROOT / "ingestion"))
sys.path.insert(0, str(PROJECT_ROOT / "retrieval"))

from chunker import chunk_documents  # noqa: E402
from embed_store import (  # noqa: E402
    DEFAULT_COLLECTION_NAME,
    DEFAULT_PERSIST_DIRECTORY,
    build_vector_store,
    delete_collection,
)
from loaders import load_documents  # noqa: E402


def main(reset: bool = False) -> None:
    """Load, chunk, embed, and persist all sample documents to Chroma.

    Args:
        reset: If True, delete the existing collection first so the rebuild
            starts from empty instead of upserting on top of it.

    Returns:
        None. Prints progress and the final collection count as it goes.
    """
    if reset:
        delete_collection()
        print(f"Deleted existing collection '{DEFAULT_COLLECTION_NAME}' (if present).")

    documents = load_documents(SAMPLE_DOCS_DIR)
    print(f"Loaded {len(documents)} document(s) from {SAMPLE_DOCS_DIR}")

    chunks = chunk_documents(documents)
    print(f"Produced {len(chunks)} chunk(s).")

    texts = [chunk.text for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    vector_store = build_vector_store(texts, metadatas)

    collection_count = vector_store._collection.count()
    print(f"Persisted collection '{DEFAULT_COLLECTION_NAME}' at {DEFAULT_PERSIST_DIRECTORY}")
    print(f"Collection count: {collection_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the HR policy Chroma index.")
    parser.add_argument(
        "--reset",
        action="store_true",
        default=False,
        help="Delete the existing persisted collection before rebuilding.",
    )
    args = parser.parse_args()
    main(reset=args.reset)
