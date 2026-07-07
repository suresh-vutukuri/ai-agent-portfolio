"""Embeds chunks with OpenAI and upserts them into an existing Pinecone index."""

from __future__ import annotations

import hashlib
import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"


def get_embeddings() -> OpenAIEmbeddings:
    """Build the OpenAI embeddings client.

    Pulls OPENAI_API_KEY from a .env file if one's present (via python-dotenv),
    otherwise falls back to whatever's already in the environment.

    Returns:
        An OpenAIEmbeddings client configured for text-embedding-3-small.

    Raises:
        RuntimeError: If OPENAI_API_KEY isn't set anywhere.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to a .env file in the project "
            "root or export it before running."
        )
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def get_index_name() -> str:
    """Read the target Pinecone index name from .env.

    Returns:
        The value of the PINECONE_INDEX_NAME environment variable.

    Raises:
        RuntimeError: If PINECONE_INDEX_NAME isn't set anywhere.
    """
    index_name = os.getenv("PINECONE_INDEX_NAME")
    if not index_name:
        raise RuntimeError(
            "PINECONE_INDEX_NAME is not set. Add it to a .env file in the "
            "project root or export it before running."
        )
    return index_name


def _make_chunk_id(text: str, metadata: dict[str, str]) -> str:
    """Hash a chunk's source, section, and text into a stable id.

    Because the id only depends on content, re-running ingestion over
    unchanged source documents upserts the same Pinecone vectors instead of
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


def build_vector_store(
    texts: list[str],
    metadatas: list[dict[str, str]],
    index_name: str | None = None,
) -> PineconeVectorStore:
    """Embed these chunks and upsert them into an existing Pinecone index.

    This never creates a Pinecone index. PineconeVectorStore.from_documents
    looks the index up by name via the Pinecone API and raises ValueError
    itself if it's not found, so there's no need to check for it here.

    Args:
        texts: Chunk text to embed, aligned index-for-index with metadatas.
        metadatas: Per-chunk metadata (source filename, section), aligned
            with texts.
        index_name: Name of the existing Pinecone index to upsert into.
            Defaults to get_index_name().

    Returns:
        The Pinecone vector store, with the given chunks embedded and
        upserted. Existing vectors with matching content-derived ids get
        overwritten rather than duplicated.

    Raises:
        ValueError: If texts and metadatas don't line up (different
            lengths), or if the target index doesn't exist.
    """
    if len(texts) != len(metadatas):
        raise ValueError(f"texts and metadatas must be the same length ({len(texts)} != {len(metadatas)})")

    documents = [Document(page_content=text, metadata=metadata) for text, metadata in zip(texts, metadatas)]
    ids = [_make_chunk_id(text, metadata) for text, metadata in zip(texts, metadatas)]

    return PineconeVectorStore.from_documents(
        documents,
        get_embeddings(),
        index_name=index_name or get_index_name(),
        ids=ids,
    )
