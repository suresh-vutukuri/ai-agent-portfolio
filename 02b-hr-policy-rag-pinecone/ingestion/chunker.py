"""Splits loaded documents into section-tagged, token-sized chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from langchain_text_splitters import RecursiveCharacterTextSplitter

from loaders import LoadedDocument

CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50
TIKTOKEN_ENCODING = "cl100k_base"
DEFAULT_SECTION_HEADING = "Preamble"

# e.g. "1. Purpose", "10. Related Policies" - a number, a period, a short title on its own line
_HEADING_PATTERN = re.compile(r"^\d{1,2}\.\s+[A-Z][^.?!]{1,78}$")


@dataclass
class Chunk:
    """A chunk of text plus enough metadata to trace it back to its source.

    Attributes:
        text: The chunk's text content.
        metadata: At minimum "source" (originating filename) and "section"
            (the heading this chunk fell under).
    """

    text: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class _Section:
    """One (heading, body) slice of a document, before it gets chunked."""

    heading: str
    text: str


def _is_heading(line: str) -> bool:
    """True if this line looks like a numbered section heading."""
    return bool(_HEADING_PATTERN.match(line.strip()))


def split_into_sections(text: str) -> list[_Section]:
    """Break a document's text into sections along its numbered headings.

    Args:
        text: Full document text, one line per paragraph/heading/table row.

    Returns:
        Sections in reading order. Anything before the first heading lands
        in a section called DEFAULT_SECTION_HEADING instead of being dropped.
    """
    sections: list[_Section] = []
    current_heading = DEFAULT_SECTION_HEADING
    current_lines: list[str] = []

    for line in text.splitlines():
        if _is_heading(line):
            if current_lines:
                sections.append(_Section(heading=current_heading, text="\n".join(current_lines)))
            current_heading = line.strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append(_Section(heading=current_heading, text="\n".join(current_lines)))

    return sections


def chunk_document(
    document: LoadedDocument,
    chunk_size: int = CHUNK_SIZE_TOKENS,
    chunk_overlap: int = CHUNK_OVERLAP_TOKENS,
) -> list[Chunk]:
    """Chunk one document section by section, tagging each chunk with its source and section.

    Args:
        document: The loaded document to chunk.
        chunk_size: Target chunk size, in tokens.
        chunk_overlap: How many tokens consecutive chunks should share.

    Returns:
        One Chunk per split, each carrying the document's filename and its
        enclosing section heading in metadata.
    """
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=TIKTOKEN_ENCODING,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks: list[Chunk] = []
    for section in split_into_sections(document.text):
        for piece in splitter.split_text(section.text):
            if not piece.strip():
                continue
            chunks.append(
                Chunk(
                    text=piece,
                    metadata={"source": document.source, "section": section.heading},
                )
            )
    return chunks


def chunk_documents(
    documents: Iterable[LoadedDocument],
    chunk_size: int = CHUNK_SIZE_TOKENS,
    chunk_overlap: int = CHUNK_OVERLAP_TOKENS,
) -> list[Chunk]:
    """Chunk a batch of documents into one flat list.

    Args:
        documents: Loaded documents to chunk.
        chunk_size: Target chunk size, in tokens.
        chunk_overlap: How many tokens consecutive chunks should share.

    Returns:
        All chunks from all documents, concatenated in the order given.
    """
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, chunk_size=chunk_size, chunk_overlap=chunk_overlap))
    return chunks
