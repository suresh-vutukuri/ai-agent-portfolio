"""Loads the persisted Chroma collection and hands back a retriever."""

from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever

from embed_store import DEFAULT_COLLECTION_NAME, DEFAULT_PERSIST_DIRECTORY, get_embeddings

TOP_K = 4


def load_vector_store(
    persist_directory: Path | str = DEFAULT_PERSIST_DIRECTORY,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> Chroma:
    """Reopen a Chroma collection that was persisted earlier.

    Args:
        persist_directory: Where the collection was persisted to.
        collection_name: Which collection to load.

    Returns:
        The loaded Chroma vector store.

    Raises:
        FileNotFoundError: If persist_directory doesn't exist, meaning the
            index hasn't been built yet.
    """
    persist_directory = Path(persist_directory)
    if not persist_directory.exists():
        raise FileNotFoundError(
            f"No persisted Chroma collection found at {persist_directory}. "
            "Run pipeline/build_index.py first."
        )
    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(persist_directory),
    )


def get_retriever(
    persist_directory: Path | str = DEFAULT_PERSIST_DIRECTORY,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    top_k: int = TOP_K,
) -> VectorStoreRetriever:
    """Build a similarity-search retriever over the persisted HR policy collection.

    Args:
        persist_directory: Where the collection was persisted to.
        collection_name: Which collection to load.
        top_k: How many chunks to return per query.

    Returns:
        A retriever doing similarity search with k=top_k.
    """
    vector_store = load_vector_store(persist_directory, collection_name)
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
