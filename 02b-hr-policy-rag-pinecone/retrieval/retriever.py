"""Connects to the existing Pinecone index and hands back a retriever."""

from __future__ import annotations

from langchain_core.vectorstores import VectorStoreRetriever
from langchain_pinecone import PineconeVectorStore

from embed_store import get_embeddings, get_index_name

TOP_K = 4


def load_vector_store(index_name: str | None = None) -> PineconeVectorStore:
    """Connect to a Pinecone index that was built earlier.

    Never creates an index - if index_name doesn't exist, the Pinecone
    client raises when it resolves the index host.

    Args:
        index_name: Name of the index to connect to. Defaults to
            get_index_name().

    Returns:
        The connected Pinecone vector store.
    """
    return PineconeVectorStore(index_name=index_name or get_index_name(), embedding=get_embeddings())


def get_retriever(
    index_name: str | None = None,
    top_k: int = TOP_K,
) -> VectorStoreRetriever:
    """Build a similarity-search retriever over the Pinecone HR policy index.

    Same interface as the Chroma version's get_retriever(top_k=...), so the
    chain and eval harness can import either implementation as a drop-in.

    Args:
        index_name: Name of the Pinecone index to query. Defaults to
            get_index_name().
        top_k: How many chunks to return per query.

    Returns:
        A retriever doing similarity search with k=top_k.
    """
    vector_store = load_vector_store(index_name)
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
