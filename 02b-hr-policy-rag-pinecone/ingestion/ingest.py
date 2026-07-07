"""Load + chunk sample_docs/ and print a summary, no embedding/storage yet."""

from __future__ import annotations

from pathlib import Path

from chunker import Chunk, chunk_documents
from loaders import load_documents

SAMPLE_DOCS_DIR = Path(__file__).resolve().parent.parent / "sample_docs"


def _print_sample_chunk(chunk: Chunk) -> None:
    """Print one chunk's metadata and text so we can eyeball it.

    Args:
        chunk: The chunk to print.

    Returns:
        None.
    """
    print("Sample chunk:")
    print(f"  metadata: {chunk.metadata}")
    print(f"  text ({len(chunk.text)} chars):")
    print(f"  {chunk.text!r}")


def main() -> None:
    """Load and chunk everything in sample_docs/, then print a quick summary.

    Returns:
        None. Results just go to stdout - this is a manual sanity check,
        nothing gets written to disk here.
    """
    documents = load_documents(SAMPLE_DOCS_DIR)
    print(f"Loaded {len(documents)} document(s) from {SAMPLE_DOCS_DIR}")
    for document in documents:
        print(f"  - {document.source} ({len(document.text)} chars)")

    chunks = chunk_documents(documents)
    print(f"\nProduced {len(chunks)} chunk(s) total.\n")

    if chunks:
        _print_sample_chunk(chunks[len(chunks) // 2])
    else:
        print("No chunks produced — check that sample_docs/ contains .docx/.pdf files.")


if __name__ == "__main__":
    main()
