"""CLI entry point: ask a question against the HR policy RAG chain.

Usage:
    python query.py "How many PTO days do I get after 4 years of service?"
"""

from __future__ import annotations

import sys

from rag_chain import build_rag_chain


def main() -> None:
    """Grab the question from argv, run it through the chain, and print the answer.

    Returns:
        None.

    Raises:
        SystemExit: If no question was passed on the command line.
    """
    if len(sys.argv) < 2:
        print('Usage: python query.py "<question>"')
        raise SystemExit(1)

    question = " ".join(sys.argv[1:])
    chain = build_rag_chain()
    answer = chain.invoke(question)
    print(answer)


if __name__ == "__main__":
    main()
