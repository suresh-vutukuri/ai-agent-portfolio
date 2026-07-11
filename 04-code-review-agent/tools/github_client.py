"""Fetches PR diffs/file contents and posts review comments via PyGithub."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict

from dotenv import load_dotenv
from github import Auth, Github, GithubException
from github.IssueComment import IssueComment

load_dotenv()


@dataclass
class PRDiff:
    """A pull request's unified diff plus the current content of its changed files.

    Attributes:
        diff_text: Unified diff covering every changed file in the PR,
            reconstructed from each file's patch.
        changed_files: Maps filename to its full file content at the PR's
            head commit. Removed files, and files whose content couldn't be
            read (e.g. binary files), are omitted.
    """

    diff_text: str
    changed_files: Dict[str, str] = field(default_factory=dict)


def _get_client() -> Github:
    """Build an authenticated PyGithub client from GITHUB_TOKEN.

    Returns:
        An authenticated Github client.

    Raises:
        RuntimeError: If GITHUB_TOKEN isn't set anywhere.
    """
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise RuntimeError(
            "GITHUB_TOKEN is not set. Add it to a .env file in the project "
            "root or export it before running."
        )
    return Github(auth=Auth.Token(token))


def fetch_pr_diff(repo_name: str, pr_number: int) -> PRDiff:
    """Fetch a pull request's diff and the current content of its changed files.

    Args:
        repo_name: Repository in "owner/name" form, e.g. "octocat/hello-world".
        pr_number: The pull request number.

    Returns:
        A PRDiff with the reconstructed unified diff text and a mapping of
        changed filename to its full content at the PR's head commit.

    Raises:
        RuntimeError: If GITHUB_TOKEN isn't set, or the repo/PR can't be
            fetched (not found, no access, rate limited, etc.).
    """
    client = _get_client()

    try:
        repo = client.get_repo(repo_name)
        pull = repo.get_pull(pr_number)
        files = list(pull.get_files())
    except GithubException as e:
        raise RuntimeError(f"Failed to fetch PR #{pr_number} from {repo_name}: {e}") from e

    diff_parts = []
    changed_files: Dict[str, str] = {}
    for pr_file in files:
        if pr_file.patch:
            old_name = pr_file.previous_filename or pr_file.filename
            diff_parts.append(f"--- a/{old_name}\n+++ b/{pr_file.filename}\n{pr_file.patch}\n")

        if pr_file.status == "removed":
            continue
        try:
            content_file = repo.get_contents(pr_file.filename, ref=pull.head.sha)
            changed_files[pr_file.filename] = content_file.decoded_content.decode("utf-8")
        except (GithubException, UnicodeDecodeError):
            continue  # binary or otherwise unreadable file; skip its content

    return PRDiff(diff_text="\n".join(diff_parts), changed_files=changed_files)


def post_review_comment(repo_name: str, pr_number: int, body: str) -> IssueComment:
    """Post a review comment to a pull request's conversation thread.

    Args:
        repo_name: Repository in "owner/name" form, e.g. "octocat/hello-world".
        pr_number: The pull request number.
        body: Markdown comment body to post.

    Returns:
        The created IssueComment.

    Raises:
        RuntimeError: If GITHUB_TOKEN isn't set, or the comment couldn't be
            posted (PR not found, no write access, etc.).
    """
    client = _get_client()

    try:
        repo = client.get_repo(repo_name)
        pull = repo.get_pull(pr_number)
        return pull.create_issue_comment(body)
    except GithubException as e:
        raise RuntimeError(f"Failed to post comment on PR #{pr_number} in {repo_name}: {e}") from e
