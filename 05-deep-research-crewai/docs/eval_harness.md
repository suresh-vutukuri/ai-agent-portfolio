# Eval Harness Methodology — Deep Research Agent (CrewAI)

## Purpose
This document describes the evaluation harness used to measure whether the deep research crew's markdown reports back their factual claims with real, supported citations rather than plausible-sounding but fabricated ones.

## Test Dataset
- **Location:** `eval/test_topics.json`
- **Size:** 5 broad research topics spanning distinct domains (energy, AI, agriculture, batteries, policy), chosen so each forces the planner/researcher/writer pipeline to run end-to-end on a fresh topic rather than a narrow variation of one.
- **Labels:** none — this eval doesn't check topical correctness against a reference answer. It checks a structural property of every report the crew produces: is each citation real and grounded.

## Metrics

### Citation Validity Score (headline metric)
Of all `[Source: URL]` citations across all reports, what fraction are both:
1. **Reachable, or only unreachable because it looks bot-blocked** — see "Bot-Blocked vs. Dead Links" below. A URL that returns a genuine 404, or fails outright (DNS/timeout/connection error), does not clear this leg.
2. **Grounded** — an LLM judge (`gpt-4o-mini` by default, see `JUDGE_MODEL` env var) scores the adjacent claim 4 or 5 out of 5 for being actually supported by the *original* Tavily search result content returned for that URL.

A citation only counts as valid if both hold. A bot-blocked URL still has to clear the judge-score bar like any other citation — it just isn't automatically failed for a reachability check that a naive HTTP client was never going to pass in the first place.

### Component Rates (diagnostic, not headline)
- `url_well_formed_rate` — fraction of citations whose URL has a valid `http(s)://` scheme and network location (catches obviously malformed/hallucinated URLs before even trying a request).
- `url_reachable_rate` — fraction of citations whose URL actually responded over HTTP (HEAD, falling back to GET on 4xx/5xx).
- `url_bot_blocked_rate` — fraction that failed but look like an anti-bot/WAF block rather than a dead link (see below).
- `url_dead_rate` — fraction that failed and look genuinely broken (404, or the request failed outright). `url_reachable_rate + url_bot_blocked_rate + url_dead_rate` always sums to 1.0 — every citation lands in exactly one bucket.
- `source_matched_rate` — fraction of citations whose URL appears in that run's own search transcript (see below) — i.e., a search the researcher agent actually ran returned it, versus the writer citing a URL from nowhere.
- `avg_judge_score` — mean 1-5 groundedness score across all citations, independent of the reachability gate.
- `avg_citations_per_report` — total citations across all reports / number of topics. A very low number here means the writer is making claims without citing them at all (see `tools/citation_tracker.uncited_lines` for that failure mode specifically).

**Groundedness matching logic:** every `web_search` call the researcher agent makes during a run is recorded to an in-memory transcript (`tools/search_tools.py`: `reset_search_transcript`/`get_search_transcript`), keyed by URL to the exact snippet Tavily returned. The judge is scored against that recorded snippet, not a fresh re-search or re-fetch of the page — this checks whether the writer accurately represented what the researcher actually saw, which is the failure mode citations are meant to prevent (a researcher's finding getting mangled or embellished in synthesis), independent of whether Tavily's own snippet was itself a good summary of the page.

### Bot-Blocked vs. Dead Links
`check_url_reachable` (in `eval/run_eval.py`) does a plain `requests` HEAD/GET — there is no headless browser behind it. That matters because a real browser and a scripted HTTP client get different answers from the same live URL:

- A **dead link** (404, or the request fails outright with a timeout/DNS failure/connection error) is genuinely broken — a browser would fail to load it too.
- A **bot-blocked link** (status 403/429/999, or a response body matching a Cloudflare-style challenge page — "just a moment", "checking your browser", "attention required", etc.) is a site that's almost certainly live and loads fine in a real browser, but sits behind a WAF/anti-bot layer that rejects a plain scripted request — via TLS/JA3 fingerprinting, a JS challenge that only a real browser can execute, or IP-reputation filtering on cloud/CI egress IPs. No amount of header-spoofing fixes this without an actual headless browser (Playwright/Selenium), which was judged too heavy a dependency for a report-only eval signal.

This was confirmed directly against `https://www.mdpi.com/...`: it returns `403 Forbidden` to this checker (with and without browser-like headers) while loading fine in an actual browser. Academic publishers behind heavy WAFs (MDPI, ScienceDirect, IEEE) are the most common source of this false negative in practice — `likely_bot_blocked=True` on one of these should be read as "our checker couldn't get in," not "this citation is fabricated."

The two buckets are mutually exclusive and cover every failed check (a malformed/non-URL string is bucketed as `likely_dead`, not bot-blocked). The classification is a heuristic, not a certainty: a WAF that resets the TCP connection instead of returning an HTTP response would look identical to a dead link's connection error here, since there's no response left to inspect for challenge markers.

## Running the Harness
```bash
python eval/run_eval.py
```
Runs each of the 5 test topics through `run_research.run_research()` (the same entry point `run_research.py`/`demo.py` use), extracts every citation from the resulting report, checks each cited URL's reachability, and scores each claim against the run's own search transcript via the LLM judge. Writes:
- `eval/results/scorecard.csv` — one row per citation (topic, claim, URL, well-formed, reachable, status code, reachability error, likely bot-blocked, likely dead, source-matched, judge score, judge rationale)
- `eval/results/summary.json` — aggregate citation validity score and the component rates above

Requires `OPENAI_API_KEY` and `TAVILY_API_KEY` in `.env` — this eval runs the full crew (real search, real LLM calls) plus real HTTP reachability checks and real judge calls, so it costs API credits and takes noticeably longer than a mocked-tool eval like `03-multiagent-triage`'s.

## CI Integration
`.github/workflows/eval-deep-research-crewai.yml` runs on every PR touching `05-deep-research-crewai/**`, using `OPENAI_API_KEY`/`TAVILY_API_KEY` from repo secrets, and posts the summary scores as a PR comment. It is **report-only** — it does not gate the PR, since a single LLM-judge run can be noisy enough (search result drift, judge variance) that hard-failing on it would create flaky red PRs rather than a reliable signal.

## Iteration Notes
A writer-prompt change was tested to see if it would raise citation validity: constrain the writer to (1) cite only the single source most directly supporting each specific claim rather than stacking multiple citations per sentence, (2) avoid combining facts from multiple sources into one synthesized claim, and (3) prefer specific, source-close claims over broad interpretive statements.

Before: `citation_validity_score` 0.545, `avg_judge_score` 3.85.
After: `citation_validity_score` 0.487, `avg_judge_score` 3.75.

The change did not measurably improve either metric — both moved slightly in the wrong direction, within what's plausibly run-to-run noise (search results and judge scoring both vary per run). Diagnostic rates (`source_matched_rate` ~98%, `url_dead_rate` near 0%) confirm the bottleneck isn't fabrication or broken links; it's that a meaningful share of claims are only loosely grounded in their cited source even when the citation itself is real. That's recorded here as a tested-and-inconclusive iteration rather than reverted or hidden, since a prompt tweak alone did not resolve it and a more effective fix (e.g., stricter judge threshold, sentence-level citation matching instead of per-line) would need further eval-design work rather than another prompt pass.

## Design Notes
- Reachability is checked with a real HTTP request (`requests`, HEAD falling back to GET, 6s timeout) rather than just format validation, since the actual failure mode being guarded against is the writer inventing a plausible-looking URL no search ever returned — format validation alone wouldn't catch that.
- The judge scores against the *recorded* search transcript rather than re-fetching the live page, so the eval is deterministic with respect to what the agent actually saw that run, and isn't penalized by pages that have since changed or gone offline (that's what `url_reachable_rate` is for, separately).
- `source_matched_rate` is tracked separately from `url_reachable_rate` because they catch different failures: a citation can be reachable (a real, live URL) but never actually appear in that run's search results (the writer swapped in a URL it recognized from training data instead of the one the researcher found) — that's a fabrication the reachability check alone would miss.
- `tools/citation_tracker.extract_citations` groups citations *per line*, not per sentence. When the writer packs multiple sentences with different citations into one paragraph (one line in the markdown), they're scored as a single "claim" spanning the whole paragraph against every URL cited anywhere in it. This is coarser than ideal — a URL that only backs the paragraph's first sentence still gets judged against the whole paragraph's text — but avoids the ambiguity of splitting a paragraph into sentences when citation markers can trail a sentence by an arbitrary number of brackets (`[Source: A][Source: B][Source: C].`).
