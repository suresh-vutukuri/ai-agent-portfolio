"""Metrics for scoring the RAG chain against the golden Q&A set."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

load_dotenv()
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4o-mini")

# matches e.g. "[Source: 01_PTO_Policy.docx, Section 3. Accrual Rates]"
_CITATION_PATTERN = re.compile(r"\[Source:\s*(?P<source>[^,\]]+),\s*Section\s*(?P<section>[^\]]+)\]")


def recall_at_k(retrieved_docs: list[Document], expected_source_doc: Optional[str]) -> Optional[float]:
    """Check whether the retriever pulled in the expected source document.

    Args:
        retrieved_docs: Chunks the retriever returned for a question.
        expected_source_doc: The golden example's expected source filename,
            without extension (e.g. "01_PTO_Policy"), or None if the
            question isn't supposed to have a source (negative controls).

    Returns:
        1.0 if any retrieved chunk's source matches (extension ignored),
        0.0 if none do, or None if there was nothing to check against.
        
    """
    if not expected_source_doc:
        return None
    for doc in retrieved_docs:
        source = doc.metadata.get("source", "")
        if Path(source).stem == expected_source_doc:
            return 1.0
    return 0.0


def citation_accuracy(answer: str, expected_source_doc: Optional[str]) -> Optional[float]:
    """Check whether every citation in the answer points to the expected source document.

    Args:
        answer: The chain's final, formatted answer.
        expected_source_doc: The golden example's expected source filename,
            without extension, or None if there's no expected source.

    Returns:
        1.0 if the answer has at least one citation and all of them name
        the expected document, 0.0 if there are no citations or a wrong
        one, or None if there was nothing to check against.
    """
    if not expected_source_doc:
        return None
    citations = _CITATION_PATTERN.findall(answer)
    if not citations:
        return 0.0
    return 1.0 if all(Path(source.strip()).stem == expected_source_doc for source, _section in citations) else 0.0


def is_refusal(answer: str, not_covered_message: str) -> bool:
    """Check whether an answer is basically the "not covered" fallback message.

    Args:
        answer: The chain's final answer.
        not_covered_message: The exact refusal sentence the prompt tells the
            model to use (see chain.prompt_template.NOT_COVERED_MESSAGE).

    Returns:
        True if the answer matches the refusal message once case and a
        trailing period are ignored, False otherwise.
    """
    normalize = lambda text: text.strip().rstrip(".").lower()
    return normalize(answer) == normalize(not_covered_message)


def _judge_score(prompt: str, llm: ChatOpenAI) -> int:
    """Send a judge prompt to the LLM and pull the 1-5 score out of the reply.

    Args:
        prompt: The full judge prompt, which asks for a single integer reply.
        llm: The judge model client.

    Returns:
        The parsed score, from 1 to 5.

    Raises:
        ValueError: If the reply doesn't contain a digit from 1 to 5.
    """
    response = llm.invoke(prompt).content
    match = re.search(r"[1-5]", str(response))
    if not match:
        raise ValueError(f"Could not parse a 1-5 score from judge response: {response!r}")
    return int(match.group())


_GROUNDEDNESS_PROMPT_TEMPLATE = """You are grading whether an AI-generated answer is fully supported \
by the provided context, on a 1-5 scale:
5 = every claim in the answer is directly supported by the context
3 = the answer is partly supported but includes some unsupported claims or overstatements
1 = the answer is largely unsupported by, or contradicts, the context

Context:
{context}

Answer:
{answer}

Reply with ONLY a single integer from 1 to 5, nothing else."""


def groundedness(context: str, answer: str, llm: ChatOpenAI) -> int:
    """Ask the LLM judge whether the answer's claims are backed by the retrieved context.

    Args:
        context: The formatted context the chain actually saw for this
            question (see chain.rag_chain.format_docs).
        answer: The chain's final answer.
        llm: The judge model client.

    Returns:
        A groundedness score from 1 (unsupported) to 5 (fully supported).
    """
    prompt = _GROUNDEDNESS_PROMPT_TEMPLATE.format(context=context, answer=answer)
    return _judge_score(prompt, llm)


_ANSWER_RELEVANCE_PROMPT_TEMPLATE = """You are grading how well an AI-generated answer matches a \
reference answer to the same question, on a 1-5 scale:
5 = conveys the same key facts as the reference answer
3 = partially correct, or missing some key facts from the reference answer
1 = contradicts the reference answer or fails to address the question

Question:
{question}

Reference answer:
{expected_answer}

AI-generated answer:
{answer}

Reply with ONLY a single integer from 1 to 5, nothing else."""


def answer_relevance(question: str, expected_answer: str, answer: str, llm: ChatOpenAI) -> int:
    """Ask the LLM judge how closely the answer matches the golden reference answer.

    Args:
        question: The original question.
        expected_answer: The golden example's reference answer.
        answer: The chain's final answer.
        llm: The judge model client.

    Returns:
        A relevance score from 1 (wrong/irrelevant) to 5 (matches the
        reference answer).
    """
    prompt = _ANSWER_RELEVANCE_PROMPT_TEMPLATE.format(
        question=question, expected_answer=expected_answer, answer=answer
    )
    return _judge_score(prompt, llm)
