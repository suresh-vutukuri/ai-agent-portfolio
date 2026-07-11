"""Main code review agent: fetch, lint, and review a PR via the Claude Agent SDK."""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    CLINotFoundError,
    ResultMessage,
    TextBlock,
    query,
)
from dotenv import load_dotenv

from review_prompt import REVIEW_SYSTEM_PROMPT

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# tools/ is a plain sibling dir, not a package
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from diff_parser import ChangedFile, parse_diff  # noqa: E402
from github_client import PRDiff, fetch_pr_diff  # noqa: E402
from lint_tools import BanditIssue, RuffIssue, run_bandit, run_ruff  # noqa: E402

DEFAULT_MODEL = "claude-sonnet-5"
MODEL = os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL)


def _lint_python_files(pr_diff: PRDiff, changed_files: List[ChangedFile]) -> Dict[str, Dict[str, list]]:
    """Run ruff and bandit on every changed Python file's current content.

    Writes each file's fetched content into a temporary directory (mirroring
    its repo-relative path, to avoid basename collisions between files in
    different directories) since the lint tools operate on real file paths,
    not in-memory content.

    Args:
        pr_diff: The fetched PR diff, whose changed_files holds each file's
            content at the PR head.
        changed_files: Parsed per-file diff entries, used to know which
            files changed.

    Returns:
        Maps each linted filename to {"ruff": [RuffIssue, ...], "bandit":
        [BanditIssue, ...]}. Non-Python files, and files with no fetched
        content (removed or binary), are skipped entirely.
    """
    findings: Dict[str, Dict[str, list]] = {}
    with tempfile.TemporaryDirectory() as tmp_dir:
        for changed_file in changed_files:
            filename = changed_file.filename
            if not filename.endswith(".py"):
                continue
            content = pr_diff.changed_files.get(filename)
            if content is None:
                continue

            tmp_path = Path(tmp_dir) / filename
            tmp_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path.write_text(content, encoding="utf-8")

            findings[filename] = {
                "ruff": run_ruff(str(tmp_path)),
                "bandit": run_bandit(str(tmp_path)),
            }
    return findings


def _format_findings(findings: Dict[str, Dict[str, list]]) -> str:
    """Render per-file lint findings as a plain-text block for the prompt.

    Args:
        findings: Output of _lint_python_files.

    Returns:
        A human-readable summary, one section per linted file.
    """
    if not findings:
        return "No changed Python files were linted (no .py files with fetched content)."

    sections = []
    for filename, tool_results in findings.items():
        lines = [f"**{filename}**"]
        ruff_issues: List[RuffIssue] = tool_results["ruff"]
        bandit_issues: List[BanditIssue] = tool_results["bandit"]

        if not ruff_issues and not bandit_issues:
            lines.append("- No ruff or bandit issues found.")
        else:
            for issue in ruff_issues:
                lines.append(f"- Ruff: line {issue.line}, {issue.rule_code}: {issue.message}")
            for issue in bandit_issues:
                lines.append(f"- Bandit: line {issue.line}, {issue.severity}: {issue.issue_text}")

        sections.append("\n".join(lines))

    return "\n\n".join(sections)


async def _run_review_query(diff_text: str, findings_text: str) -> str:
    """Send the diff and lint findings to Claude and collect the final review text.

    Uses the Claude Agent SDK's query() for a tool-free request: allowed_tools=[]
    means Claude reasons only over the prompt text (it never tries to read the
    repo itself). max_turns=8 gives Claude room for its own internal turns
    (e.g. planning before the final answer) - max_turns=1 was too tight and
    the query would fail with "Reached maximum number of turns (1)" before
    producing a review.

    Args:
        diff_text: The PR's unified diff.
        findings_text: Formatted ruff/bandit findings for changed Python files.

    Returns:
        The reviewer's final markdown response.

    Raises:
        RuntimeError: If the Claude Code CLI isn't installed, the query
            errors out (auth, billing, rate limit, or an unsuccessful
            ResultMessage), or it produces no text.
    """
    prompt = (
        "## Unified Diff\n\n"
        f"```diff\n{diff_text}\n```\n\n"
        "## Linter Findings\n\n"
        f"{findings_text}"
    )

    options = ClaudeAgentOptions(
        system_prompt=REVIEW_SYSTEM_PROMPT,
        model=MODEL,
        allowed_tools=[],
        max_turns=8,
    )

    review_text = ""
    # query() is an async generator; breaking out of `async for` early (once
    # we have our ResultMessage) does NOT close it - `async for` never calls
    # aclose() on early exit (PEP 533 was deferred; the SDK's own internals
    # rely on an explicit aclose() for exactly this reason). Without an
    # explicit aclose() here, the underlying CLI subprocess only gets torn
    # down via asyncio's implicit generator-shutdown path when this
    # coroutine's event loop closes, which races with cleanup across the
    # repeated asyncio.run() calls an eval harness makes and raises
    # "RuntimeError: aclose(): asynchronous generator is already running".
    response_stream = query(prompt=prompt, options=options)
    try:
        async for message in response_stream:
            if isinstance(message, AssistantMessage):
                if message.error:
                    raise RuntimeError(f"Claude Agent SDK query failed: {message.error}")
                for block in message.content:
                    if isinstance(block, TextBlock):
                        review_text += block.text
            elif isinstance(message, ResultMessage):
                if message.is_error:
                    raise RuntimeError(f"Claude Agent SDK query failed: {message.errors or message.result}")
                break
    except CLINotFoundError as e:
        raise RuntimeError(
            "The Claude Code CLI is required by the Claude Agent SDK but wasn't found. "
            "Install it with `npm install -g @anthropic-ai/claude-code`."
        ) from e
    finally:
        await response_stream.aclose()

    if not review_text:
        raise RuntimeError("Claude Agent SDK query returned no text.")

    return review_text


def review_pr(repo_name: str, pr_number: int) -> str:
    """Fetch, lint, and review a pull request, returning a markdown review.

    Does not post anything back to GitHub - see tools/github_client.py's
    post_review_comment for that separate, explicit step.

    Args:
        repo_name: Repository in "owner/name" form, e.g. "octocat/hello-world".
        pr_number: The pull request number.

    Returns:
        A structured markdown code review, grouped by severity.

    Raises:
        RuntimeError: If GITHUB_TOKEN isn't set, the PR can't be fetched, the
            Claude Code CLI isn't installed, or the Claude Agent SDK query fails.
    """
    pr_diff = fetch_pr_diff(repo_name, pr_number)
    changed_files = parse_diff(pr_diff.diff_text)

    findings = _lint_python_files(pr_diff, changed_files)
    findings_text = _format_findings(findings)

    return asyncio.run(_run_review_query(pr_diff.diff_text, findings_text))
