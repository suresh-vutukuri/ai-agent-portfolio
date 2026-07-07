"""Runs the golden Q&A set through the RAG chain and scores each answer.

Negative-control questions have no expected source/answer, so we just check
whether the chain correctly refused instead of guessing.

Writes eval/results/scorecard.csv (per-question) and eval/results/summary.json
(aggregate averages).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from langchain_core.vectorstores import VectorStoreRetriever

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = Path(__file__).resolve().parent
GOLDEN_SET_PATH = EVAL_DIR / "golden_qa_eval_set.json"
RESULTS_DIR = EVAL_DIR / "results"

sys.path.insert(0, str(PROJECT_ROOT / "retrieval"))
sys.path.insert(0, str(PROJECT_ROOT / "chain"))

from retriever import get_retriever  # noqa: E402
from rag_chain import build_rag_chain, format_docs  # noqa: E402
from prompt_template import NOT_COVERED_MESSAGE  # noqa: E402

from metrics import (  # noqa: E402
    JUDGE_MODEL,
    answer_relevance,
    citation_accuracy,
    groundedness,
    is_refusal,
    recall_at_k,
)

NEGATIVE_CONTROL_CATEGORY = "negative_control"
TOP_K = 4

SCORECARD_FIELDNAMES = [
    "id",
    "category",
    "question",
    "source_doc",
    "answer",
    "expected_answer",
    "recall_at_k",
    "groundedness",
    "citation_accuracy",
    "answer_relevance",
    "refusal_correct",
]


def _load_golden_set(path: Path) -> list[dict[str, Any]]:
    """Read the golden Q&A examples from JSON.

    Args:
        path: Path to golden_qa_eval_set.json.

    Returns:
        The list of example dicts, in file order.
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_example(
    example: dict[str, Any],
    retriever: VectorStoreRetriever,
    rag_chain: Runnable[str, str],
    judge_llm: ChatOpenAI,
) -> dict[str, Any]:
    """Run one golden example through the chain and score it.

    Negative-control rows skip the numeric metrics entirely and just check
    for a correct refusal instead, since there's no expected source/answer
    to compare against.

    Args:
        example: One entry from golden_qa_eval_set.json.
        retriever: Retriever used to check recall@k against the same chunks
            the chain pulls in internally.
        rag_chain: The end-to-end chain being evaluated.
        judge_llm: Shared LLM-judge client for groundedness/relevance scoring.

    Returns:
        A row dict matching SCORECARD_FIELDNAMES, with None for whichever
        metrics don't apply to this row.
    """
    question: str = example["question"]
    source_doc: Optional[str] = example.get("source_doc")
    category: str = example.get("category", "")
    expected_answer: str = example.get("expected_answer", "")

    retrieved_docs: list[Document] = retriever.invoke(question)
    answer: str = rag_chain.invoke(question)

    row: dict[str, Any] = {
        "id": example["id"],
        "category": category,
        "question": question,
        "source_doc": source_doc,
        "answer": answer,
        "expected_answer": expected_answer,
        "recall_at_k": None,
        "groundedness": None,
        "citation_accuracy": None,
        "answer_relevance": None,
        "refusal_correct": None,
    }

    if category == NEGATIVE_CONTROL_CATEGORY:
        row["refusal_correct"] = is_refusal(answer, NOT_COVERED_MESSAGE)
    else:
        context = format_docs(retrieved_docs)
        row["recall_at_k"] = recall_at_k(retrieved_docs, source_doc)
        row["groundedness"] = groundedness(context, answer, judge_llm)
        row["citation_accuracy"] = citation_accuracy(answer, source_doc)
        row["answer_relevance"] = answer_relevance(question, expected_answer, answer, judge_llm)

    return row


def _write_scorecard_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Dump one row per question to a CSV file.

    Args:
        rows: Per-question result dicts, as produced by evaluate_example.
        path: Where to write the CSV; parent directories get created if missing.

    Returns:
        None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SCORECARD_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: list[float]) -> Optional[float]:
    """Average a list of numbers.

    Args:
        values: Numbers to average.

    Returns:
        The mean, or None if the list is empty.
    """
    return sum(values) / len(values) if values else None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Roll the per-question rows up into aggregate averages.

    Args:
        rows: Per-question result dicts, as produced by evaluate_example.

    Returns:
        A dict with mean recall@k, groundedness, citation accuracy, and
        answer relevance across the scored (non-negative-control)
        questions, plus negative-control refusal accuracy and question counts.
    """
    scored_rows = [row for row in rows if row["category"] != NEGATIVE_CONTROL_CATEGORY]
    negative_rows = [row for row in rows if row["category"] == NEGATIVE_CONTROL_CATEGORY]

    return {
        "num_questions": len(rows),
        "num_scored": len(scored_rows),
        "num_negative_control": len(negative_rows),
        "recall_at_k": _mean([row["recall_at_k"] for row in scored_rows]),
        "groundedness": _mean([row["groundedness"] for row in scored_rows]),
        "citation_accuracy": _mean([row["citation_accuracy"] for row in scored_rows]),
        "answer_relevance": _mean([row["answer_relevance"] for row in scored_rows]),
        "negative_control_refusal_accuracy": _mean(
            [1.0 if row["refusal_correct"] else 0.0 for row in negative_rows]
        ),
    }


def main() -> None:
    """Evaluate the whole golden set and write the scorecard + summary.

    Returns:
        None. Writes eval/results/scorecard.csv and eval/results/summary.json,
        and prints the summary to stdout as it goes.
    """
    load_dotenv()

    golden_set = _load_golden_set(GOLDEN_SET_PATH)
    print(f"Loaded {len(golden_set)} golden example(s) from {GOLDEN_SET_PATH}")

    retriever = get_retriever(top_k=TOP_K)
    rag_chain = build_rag_chain(top_k=TOP_K)
    judge_llm = ChatOpenAI(model=JUDGE_MODEL, temperature=0)

    rows = []
    for example in golden_set:
        print(f"  Evaluating {example['id']}...")
        rows.append(evaluate_example(example, retriever, rag_chain, judge_llm))

    scorecard_path = RESULTS_DIR / "scorecard.csv"
    summary_path = RESULTS_DIR / "summary.json"

    _write_scorecard_csv(rows, scorecard_path)
    summary = summarize(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nWrote {scorecard_path}")
    print(f"Wrote {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
