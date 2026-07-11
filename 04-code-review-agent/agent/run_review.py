"""CLI: review a GitHub pull request with Claude, and optionally post the review.

Usage:
    python run_review.py --repo owner/repo --pr 12
    python run_review.py --repo owner/repo --pr 12 --post
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from reviewer import review_pr

# tools/ is a plain sibling dir, not a package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from github_client import post_review_comment  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse --repo, --pr, and --post from the command line.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Review a GitHub pull request with Claude.")
    parser.add_argument("--repo", required=True, help='Repository in "owner/name" form.')
    parser.add_argument("--pr", required=True, type=int, help="Pull request number.")
    parser.add_argument(
        "--post",
        action="store_true",
        help="Post the review as a PR comment. Without this flag, the review is only printed.",
    )
    return parser.parse_args()


def main() -> None:
    """Review the given PR, print it, and post it only if --post was passed.

    Returns:
        None.
    """
    args = parse_args()
    review = review_pr(args.repo, args.pr)
    print(review)

    if args.post:
        post_review_comment(args.repo, args.pr, review)
        print(f"\nPosted review comment to {args.repo}#{args.pr}.")


if __name__ == "__main__":
    main()
