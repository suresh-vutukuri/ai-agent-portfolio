"""Citation-validity eval: are the report's [Source: URL] citations real and supported?

Runs each test topic through run_research.run_research(), then for every citation
the writer produced (see tools/citation_tracker.extract_citations) checks two
independent things:
  (a) URL validity - is the cited URL well-formed and actually reachable over
      HTTP, or does it look fabricated/dead. A failed check is further
      bucketed into likely_bot_blocked vs. likely_dead (see
      check_url_reachable), since a naive HTTP client with no headless
      browser will read plenty of live, WAF-protected pages (MDPI,
      ScienceDirect, IEEE, etc.) as "unreachable" even though they're fine.
  (b) Groundedness - an LLM judge (gpt-4o-mini by default) scores 1-5 whether
      the claim next to the citation is actually supported by the *original*
      Tavily search result content returned for that URL. That original
      content is recovered from the run's search transcript (see
      tools/search_tools.get_search_transcript), not re-fetched or re-searched,
      so the judge is scored against exactly what the researcher agent saw.

Writes eval/results/scorecard.csv (one row per citation) and
eval/results/summary.json (aggregate citation validity and citations/report).
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = Path(__file__).resolve().parent
TEST_TOPICS_PATH = EVAL_DIR / "test_topics.json"
RESULTS_DIR = EVAL_DIR / "results"

# tools/ is a plain sibling dir, not a package
sys.path.insert(0, str(PROJECT_ROOT / "tools"))
sys.path.insert(0, str(PROJECT_ROOT))

from citation_tracker import extract_citations  # noqa: E402
from search_tools import build_url_content_index, get_search_transcript, reset_search_transcript  # noqa: E402
from run_research import run_research  # noqa: E402

load_dotenv()
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4o-mini")
REQUEST_TIMEOUT = 6  # seconds, for URL reachability checks
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; DeepResearchEvalBot/1.0)"}

# Status codes that are themselves a strong bot/WAF-block signal, independent
# of response body content: 403 (generic block), 429 (rate limited), 999
# (LinkedIn's dedicated anti-scraping status).
_BOT_BLOCK_STATUS_CODES = {403, 429, 999}

# Substrings (checked case-insensitively) that show up in Cloudflare/other WAF
# interstitial/challenge pages - used to catch a block riding on an otherwise
# ambiguous status code (e.g. a 503 "Just a moment..." JS-challenge page,
# which looks identical to a genuine server outage by status code alone).
_WAF_CHALLENGE_MARKERS = (
    "checking your browser",
    "attention required",
    "cf-browser-verification",
    "just a moment",
    "please enable cookies",
    "sorry, you have been blocked",
    "verify you are human",
    "unusual traffic",
    "request blocked",
)

SCORECARD_FIELDNAMES = [
    "topic_id",
    "topic",
    "claim",
    "url",
    "well_formed",
    "reachable",
    "status_code",
    "reachability_error",
    "likely_bot_blocked",
    "likely_dead",
    "source_matched",
    "judge_score",
    "judge_rationale",
]

_JUDGE_PROMPT_TEMPLATE = """You are grading whether a factual claim from a research report is \
supported by the web search result content it was cited from, on a 1-5 scale:
5 = the claim is fully and directly supported by the source content
3 = the source content is related but only partially supports the claim, or the claim overstates it
1 = the source content does not support the claim at all, or no source content was found

Claim:
{claim}

Original search result content for the cited URL:
{source_content}

Reply with ONLY a JSON object of the form {{"score": <1-5 integer>, "rationale": "<one sentence>"}}."""


def _load_test_topics(path: Path = TEST_TOPICS_PATH) -> list[dict[str, Any]]:
    """Read the test topics from JSON.

    Args:
        path: Path to test_topics.json.

    Returns:
        The list of topic dicts (id, topic), in file order.
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _peek_body_text(response: requests.Response, max_bytes: int = 4096) -> str:
    """Read a small prefix of a response body as text, without downloading it all.

    Used only to sniff for WAF-challenge-page markers; reading the full body
    of a large PDF or video page just to classify a 403 would be wasteful.

    Args:
        response: An in-flight `requests` response (may be `stream=True`).
        max_bytes: Maximum number of bytes to read from the body.

    Returns:
        The decoded text of the first `max_bytes` of the body, or "" if the
        body couldn't be read (e.g. already closed, or a decode failure).
    """
    try:
        chunk = next(response.iter_content(chunk_size=max_bytes), b"")
    except Exception:
        return ""
    try:
        return chunk.decode(response.encoding or "utf-8", errors="ignore")
    except (LookupError, TypeError):
        return chunk.decode("utf-8", errors="ignore")


def _classify_unreachable(status_code: Optional[int], body_sample: str) -> tuple[bool, bool]:
    """Bucket a failed request as a likely bot/WAF block vs. a likely dead link.

    The two buckets are mutually exclusive and exhaustive for any unreachable
    URL: either the link itself looks broken (dead), or it looks live but
    something between us and it is blocking automated requests specifically
    (bot-blocked) - see the module docstring and docs/eval_harness.md for why
    that distinction matters for a naive (non-headless-browser) HTTP check.

    Args:
        status_code: The HTTP status observed, or None if the request failed
            outright (timeout/DNS failure/connection error - always dead,
            since there was no response to have been a soft WAF block).
        body_sample: A short text sample of the response body (may be empty).

    Returns:
        (likely_bot_blocked, likely_dead).
    """
    if status_code is None:
        return False, True

    has_challenge_markers = any(marker in body_sample.lower() for marker in _WAF_CHALLENGE_MARKERS)
    if status_code in _BOT_BLOCK_STATUS_CODES or has_challenge_markers:
        return True, False
    return False, True


def check_url_reachable(url: str) -> dict[str, Any]:
    """Check whether a cited URL is well-formed and actually reachable over HTTP.

    Tries a HEAD request first, falling back to GET if the server rejects HEAD
    (some sites return 403/405 for HEAD specifically). Network failures
    (timeout, DNS failure, connection refused) count as unreachable rather
    than raising, since a dead/fabricated URL is exactly what this checks for.

    A failed request is further bucketed into "likely_bot_blocked" (the site
    is almost certainly live, but a WAF/anti-bot layer is rejecting a plain
    HTTP client - 403/429/999 or a Cloudflare-style challenge page) vs.
    "likely_dead" (404, a genuine connection failure, or an unrecognized
    error) - see docs/eval_harness.md for why this matters: without a
    headless browser, JS-challenge-protected publishers (MDPI, ScienceDirect,
    IEEE, etc.) will always read as unreachable here even when they're live.

    Args:
        url: The cited URL to check.

    Returns:
        A dict with:
          - "well_formed": True if the URL has an http(s) scheme and a network location.
          - "reachable": True if a request returned a non-error status (<400),
            False if it returned an error status or the request failed
            outright, or None if the URL wasn't well-formed enough to try.
          - "status_code": The HTTP status code observed, or None.
          - "error": A short, human-readable reason the URL was unreachable
            (the HTTP status line, or the requests exception type/message for
            a network-level failure), or None if it was reachable.
          - "likely_bot_blocked": True if the failure looks like an anti-bot/WAF
            block rather than a dead link. Always False when reachable.
          - "likely_dead": True if the failure looks like a genuinely broken
            link (not found, or unreachable outright). Always False when
            reachable. Mutually exclusive with likely_bot_blocked.
    """
    parsed = urlparse(url)
    well_formed = parsed.scheme in ("http", "https") and bool(parsed.netloc)
    if not well_formed:
        return {
            "well_formed": False,
            "reachable": None,
            "status_code": None,
            "error": "not a well-formed http(s) URL",
            "likely_bot_blocked": False,
            "likely_dead": True,
        }

    try:
        response = requests.head(url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        if response.status_code >= 400:
            response = requests.get(
                url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT, allow_redirects=True, stream=True
            )
        reachable = response.status_code < 400
        if reachable:
            return {
                "well_formed": True,
                "reachable": True,
                "status_code": response.status_code,
                "error": None,
                "likely_bot_blocked": False,
                "likely_dead": False,
            }

        body_sample = _peek_body_text(response)
        bot_blocked, dead = _classify_unreachable(response.status_code, body_sample)
        return {
            "well_formed": True,
            "reachable": False,
            "status_code": response.status_code,
            "error": f"HTTP {response.status_code} {response.reason}",
            "likely_bot_blocked": bot_blocked,
            "likely_dead": dead,
        }
    except requests.RequestException as exc:
        return {
            "well_formed": True,
            "reachable": False,
            "status_code": None,
            "error": f"{type(exc).__name__}: {exc}",
            "likely_bot_blocked": False,
            "likely_dead": True,
        }


def judge_citation_support(claim: str, source_content: Optional[str], client: OpenAI) -> tuple[int, str]:
    """Ask the LLM judge whether a claim is supported by the source's original content.

    Args:
        claim: The report's claim text (citation marker already stripped).
        source_content: The snippet Tavily originally returned for the cited
            URL, or None if that URL never appeared in the run's search
            transcript (i.e. the writer cited a URL no search actually returned).
        client: The OpenAI client used to call the judge model.

    Returns:
        A (score, rationale) tuple; score is 1 (unsupported/fabricated) to 5
        (fully supported). Falls back to a regex-extracted digit and the raw
        reply text if the judge doesn't return valid JSON.
    """
    content = source_content if source_content else "(No original search result content was found for this URL.)"
    prompt = _JUDGE_PROMPT_TEMPLATE.format(claim=claim, source_content=content)
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    raw = response.choices[0].message.content or ""
    try:
        parsed = json.loads(raw)
        return int(parsed["score"]), str(parsed.get("rationale", ""))
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        match = re.search(r"[1-5]", raw)
        score = int(match.group()) if match else 1
        return score, raw.strip()


def evaluate_topic(topic_id: str, topic: str, client: OpenAI) -> list[dict[str, Any]]:
    """Run one topic through the research crew and score every citation in its report.

    Args:
        topic_id: The topic's short id (e.g. "T1"), carried through to each row.
        topic: The research topic to investigate.
        client: The OpenAI client used for LLM-judge scoring.

    Returns:
        One row dict (matching SCORECARD_FIELDNAMES) per cited URL found in
        the report. Empty if the report contained no citations.
    """
    reset_search_transcript()
    report = run_research(topic)
    url_content_index = build_url_content_index(get_search_transcript())

    rows: list[dict[str, Any]] = []
    for citation in extract_citations(report):
        claim = citation["claim"]
        for url in citation["sources"]:
            reachability = check_url_reachable(url)
            if not reachability["reachable"]:
                bucket = "bot-blocked" if reachability["likely_bot_blocked"] else "dead"
                print(
                    f"    [unreachable:{bucket}] {url} "
                    f"status={reachability['status_code']} error={reachability['error']}"
                )
            source_content = url_content_index.get(url)
            score, rationale = judge_citation_support(claim, source_content, client)
            rows.append(
                {
                    "topic_id": topic_id,
                    "topic": topic,
                    "claim": claim,
                    "url": url,
                    "well_formed": reachability["well_formed"],
                    "reachable": reachability["reachable"],
                    "status_code": reachability["status_code"],
                    "reachability_error": reachability["error"],
                    "likely_bot_blocked": reachability["likely_bot_blocked"],
                    "likely_dead": reachability["likely_dead"],
                    "source_matched": url in url_content_index,
                    "judge_score": score,
                    "judge_rationale": rationale,
                }
            )
    return rows


def _write_scorecard_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Dump one row per citation to a CSV file.

    Args:
        rows: Per-citation result dicts, as produced by evaluate_topic.
        path: Where to write the CSV; parent directories get created if missing.

    Returns:
        None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SCORECARD_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: list[Optional[float]]) -> Optional[float]:
    """Average the non-None values in a list.

    Args:
        values: Numbers (or None) to average.

    Returns:
        The mean of the non-None values, or None if there are none.
    """
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def summarize(rows: list[dict[str, Any]], num_topics: int) -> dict[str, Any]:
    """Roll the per-citation rows up into aggregate citation-validity metrics.

    A citation counts as "valid" if the judge scored it 4 or 5 (the claim is
    actually backed by the original source content) AND the URL either
    checked out as reachable OR was only flagged unreachable because it looks
    bot-blocked (see check_url_reachable / docs/eval_harness.md). Bot-blocked
    URLs get a *soft pass* on the reachability leg of the gate rather than an
    automatic fail, since the link itself likely isn't broken - our checker
    just couldn't get past its WAF. They still must clear the judge-score bar
    like any other citation, so a bot-blocked URL with a low judge score
    (e.g. no matching search-transcript content) is not rescued by this.

    citation_validity_score is the headline metric; the component rates
    (well-formed/reachable/bot-blocked/dead/source-matched/avg judge score)
    are kept alongside for diagnosing *why* it's low. url_reachable_rate,
    url_bot_blocked_rate, and url_dead_rate partition every citation into
    exactly one bucket, so they sum to 1.0.

    Args:
        rows: Per-citation result dicts across all topics, as produced by evaluate_topic.
        num_topics: How many topics were run, for avg_citations_per_report.

    Returns:
        A dict of aggregate counts and rates.
    """
    total_citations = len(rows)
    num_valid = sum(1 for r in rows if (r["reachable"] or r["likely_bot_blocked"]) and r["judge_score"] >= 4)

    return {
        "num_topics": num_topics,
        "total_citations": total_citations,
        "avg_citations_per_report": total_citations / num_topics if num_topics else None,
        "citation_validity_score": num_valid / total_citations if total_citations else None,
        "url_well_formed_rate": _mean([1.0 if r["well_formed"] else 0.0 for r in rows]),
        "url_reachable_rate": _mean([1.0 if r["reachable"] else 0.0 for r in rows]),
        "url_bot_blocked_rate": _mean([1.0 if r["likely_bot_blocked"] else 0.0 for r in rows]),
        "url_dead_rate": _mean([1.0 if r["likely_dead"] else 0.0 for r in rows]),
        "source_matched_rate": _mean([1.0 if r["source_matched"] else 0.0 for r in rows]),
        "avg_judge_score": _mean([float(r["judge_score"]) for r in rows]),
    }


def main() -> None:
    """Evaluate every test topic and write the scorecard + summary.

    Returns:
        None. Writes eval/results/scorecard.csv and eval/results/summary.json,
        and prints the summary to stdout as it goes.
    """
    topics = _load_test_topics()
    print(f"Loaded {len(topics)} test topic(s) from {TEST_TOPICS_PATH}")

    client = OpenAI()

    rows: list[dict[str, Any]] = []
    for entry in topics:
        print(f"  Evaluating {entry['id']}: {entry['topic']!r}...")
        topic_rows = evaluate_topic(entry["id"], entry["topic"], client)
        print(f"    {len(topic_rows)} citation(s) found and scored")
        rows.extend(topic_rows)

    scorecard_path = RESULTS_DIR / "scorecard.csv"
    summary_path = RESULTS_DIR / "summary.json"

    _write_scorecard_csv(rows, scorecard_path)
    summary = summarize(rows, num_topics=len(topics))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nWrote {scorecard_path}")
    print(f"Wrote {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
