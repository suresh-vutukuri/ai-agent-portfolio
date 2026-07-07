"""LCEL RAG chain: retriever | prompt | gpt-4o-mini | citation parser."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from re import Match

from langchain_core.documents import Document
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI

from prompt_template import RAG_PROMPT

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "retrieval"))

from retriever import get_retriever  # noqa: E402

LLM_MODEL = "gpt-4o-mini"

# Using [[ ]] instead of ( ) here because several section headings contain
# literal parens (e.g. "9. Flexible PTO (Director+)"), which broke an
# earlier version of this regex.
_CITATION_TAG_PATTERN = re.compile(r"\[\[cite:\s*(?P<source>[^|]+?)\s*\|\s*(?P<section>.+?)\]\]")


class CitationFormattingOutputParser(BaseOutputParser[str]):
    """Rewrites the model's raw citation tags into the final display format.

    The prompt asks the model to tag claims with "[[cite: source | section]]"
    since that's easier for it to produce reliably. This parser turns each
    tag into "[Source: source, Section section]" for the actual output;
    anything without a tag (like the "not covered" fallback) passes through
    untouched.
    """

    def parse(self, text: str) -> str:
        """Swap every raw citation tag in the text for its display form.

        Args:
            text: Raw text from the LLM, possibly containing "[[cite: ...]]" tags.

        Returns:
            The same text with each tag replaced by "[Source: ..., Section ...]".
        """

        def _replace(match: Match[str]) -> str:
            source = match.group("source").strip()
            section = match.group("section").strip()
            return f"[Source: {source}, Section {section}]"

        return _CITATION_TAG_PATTERN.sub(_replace, text).strip()

    @property
    def _type(self) -> str:
        """Name LangChain uses to identify this parser when serialized."""
        return "citation_formatting_output_parser"


def format_docs(docs: list[Document]) -> str:
    """Wrap each retrieved chunk in an <excerpt> tag carrying its source/section.

    Giving the model an exact source/section value to copy (rather than
    making it reconstruct one) is what keeps its citations accurate.

    Args:
        docs: Chunks the retriever returned for a question.

    Returns:
        One <excerpt source=".." section="..">...</excerpt> block per chunk,
        joined with blank lines in between.
    """
    blocks = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown source")
        section = doc.metadata.get("section", "unknown section")
        blocks.append(f'<excerpt source="{source}" section="{section}">\n{doc.page_content}\n</excerpt>')
    return "\n\n".join(blocks)


def build_rag_chain(top_k: int = 4) -> Runnable[str, str]:
    """Assemble the question -> cited-answer chain.

    Args:
        top_k: How many chunks the retriever should pull in per question.

    Returns:
        A runnable that takes a question string and returns the final,
        citation-formatted answer string.
    """
    retriever = get_retriever(top_k=top_k)
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

    return (
        RunnableParallel(context=retriever | format_docs, question=RunnablePassthrough())
        | RAG_PROMPT
        | llm
        | CitationFormattingOutputParser()
    )
