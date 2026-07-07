"""Embeds chunks with OpenAI and stores them in a persisted Chroma collection."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_PERSIST_DIRECTORY = Path(__file__).resolve().parent.parent / "db" / "chroma_db"
DEFAULT_COLLECTION_NAME = "hr_policies"


def get_embeddings() -> OpenAIEmbeddings:
    """Build the OpenAI embeddings client.

    Pulls OPENAI_API_KEY from a .env file if one's present (via python-dotenv),
    otherwise falls back to whatever's already in the environment.

    Returns:
        An OpenAIEmbeddings client configured for text-embedding-3-small.

    Raises:
        RuntimeError: If OPENAI_API_KEY isn't set anywhere.
    """
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to a .env file in the project "
            "root or export it before running."
        )
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def _make_chunk_id(text: str, metadata: dict[str, str]) -> str:
    """Hash a chunk's source, section, and text into a stable id.

    Because the id only depends on content, re-running ingestion over
    unchanged source documents upserts the same Chroma entries instead of
    piling up duplicates. Only chunks whose text or section actually
    changed end up looking "new".

    Args:
        text: The chunk's text.
        metadata: The chunk's metadata dict; expected to have "source" and
            "section" keys.

    Returns:
        A hex SHA-256 digest that uniquely identifies this chunk.
    """
    key = f"{metadata.get('source', '')}|{metadata.get('section', '')}|{text}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def delete_collection(
    persist_directory: Path | str = DEFAULT_PERSIST_DIRECTORY,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> None:
    """Delete a persisted collection if it exists.

    Meant for resetting the index before a full rebuild, so chunks left
    over from edited or removed source docs don't linger next to freshly
    re-embedded ones. Talks to chromadb directly rather than going through
    a langchain Chroma instance, so it doesn't need an embeddings client
    (and therefore no API key) just to delete something.

    Args:
        persist_directory: Where the Chroma collection lives on disk.
        collection_name: Which collection to delete.

    Returns:
        None. Does nothing if the directory or collection doesn't exist.
    """
    persist_directory = Path(persist_directory)
    if not persist_directory.exists():
        return

    client = chromadb.PersistentClient(path=str(persist_directory))
    existing_names = {collection.name for collection in client.list_collections()}
    if collection_name in existing_names:
        client.delete_collection(collection_name)


def build_vector_store(
    texts: list[str],
    metadatas: list[dict[str, str]],
    persist_directory: Path | str = DEFAULT_PERSIST_DIRECTORY,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> Chroma:
    """Create (or reopen) the persisted collection and upsert these chunks into it.

    Args:
        texts: Chunk text to embed, aligned index-for-index with metadatas.
        metadatas: Per-chunk metadata (source filename, section), aligned
            with texts.
        persist_directory: Where Chroma should persist the collection.
        collection_name: Name of the collection to create or add to.

    Returns:
        The Chroma vector store, with the given chunks embedded and persisted.
        Existing entries with matching content-derived ids get overwritten
        rather than duplicated.

    Raises:
        ValueError: If texts and metadatas don't line up (different lengths).
    """
    if len(texts) != len(metadatas):
        raise ValueError(f"texts and metadatas must be the same length ({len(texts)} != {len(metadatas)})")

    embeddings = get_embeddings()

    persist_directory = Path(persist_directory)
    persist_directory.mkdir(parents=True, exist_ok=True)

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )

    documents = [Document(page_content=text, metadata=metadata) for text, metadata in zip(texts, metadatas)]
    ids = [_make_chunk_id(text, metadata) for text, metadata in zip(texts, metadatas)]
    vector_store.add_documents(documents, ids=ids)
    return vector_store
