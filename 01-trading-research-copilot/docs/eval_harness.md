# Eval Harness Methodology — Trading Research Copilot

## Purpose
This document describes the evaluation harness used to measure whether the HTF (1-hour) bias engine's computed direction agrees with a human reviewer's visual read of the chart, and whether the LTF (5-minute) entry-candidate output is structurally sound.

## Golden Dataset
- **Location:** `eval/golden_dataset.json`
- **Size:** 5 entries across two instruments
  - 3 ES=F dates, 2 NQ=F dates
  - A mix of trending (bullish, bearish) and choppy/range-bound periods
- Each entry is `{date, ticker, expected_bias, notes}`:
  - `date` — the historical cutoff to compute the bias "as of" (via `htf_data.fetch_htf_bars(..., end_date=date)`, which fetches only bars up to end-of-day on that date, so nothing after the cutoff leaks into the computation).
  - `expected_bias` — `bullish` / `bearish` / `neutral`, a human's visual-review label based on the shape of daily closes leading up to `date` (higher-highs/higher-lows vs. lower-highs/lower-lows vs. sideways chop) - **not** derived by running `compute_htf_bias()` itself and copying its answer, since that would make the eval circular.
  - `notes` — the specific price levels/dates that motivated the label, so the reasoning is auditable rather than an unexplained tag.

## Metrics

### 1. Bias Accuracy (headline metric)
For each golden entry, fetch 1H bars ending at `date` and run `compute_htf_bias()`. Compare the resulting `bias` field to `expected_bias`. Binary per entry (1 = match, 0 = mismatch), averaged across all entries.

**What it measures:** whether the BOS/CHoCH-driven bias call agrees with an independent human read of the same chart.

### 2. Candidate Sanity-Check Rate
For each golden entry, take the top-ranked **near-term** candidate (if any) from `find_entry_candidates()` and check two structural properties:
- **Invalidation level exists** — derivable from the zone itself (bottom of a bullish zone, top of a bearish zone) and the zone is well-formed (`top > bottom`).
- **Within near-term threshold** — the candidate's own `distance_from_current_price['pct']` is actually `<= DEFAULT_NEAR_TERM_THRESHOLD_PCT`, catching a tiering/ranking bug rather than a data problem.

Entries with no near-term candidate at all are counted separately as "skipped," not as failures - the pass rate is `passed / checked`, where `checked` excludes skips.

**What it measures:** structural correctness of the candidate output (a well-formed zone, correctly tiered) - not whether the candidate would have been a profitable trade.

## Running the Harness
```bash
python eval/run_eval.py
```
For each golden entry, this fetches point-in-time HTF data (via `end_date`), computes the bias, then runs the candidate-sanity check (see Limitations below for what LTF data that check actually uses), and writes:
- `eval/results/summary.json` — bias accuracy, candidate sanity-check rate, and the full per-entry breakdown (computed bias, confidence, top candidate, sanity-check detail, any errors)

This harness makes no LLM/API calls of its own — it only exercises the deterministic analysis pipeline (`fetch_htf_bars`, `compute_htf_bias`, `find_entry_candidates`), so it's free and repeatable and requires no secrets to run. It does make live `yfinance` network calls per entry, so a run takes a few seconds per instrument/date and can occasionally hit `yfinance` rate limits (handled by `data_utils.retry_on_empty`).

## Latest Results
| Metric | Score |
|---|---|
| Bias Accuracy | 80.0% (4/5) |
| Candidate Sanity-Check Rate | 100.0% (5/5 checked, 0 skipped) |

The one mismatch: the ES=F 2026-06-26 entry was hand-labeled `neutral` (a choppy, range-bound week and a half after the mid-June high, with repeated failed bounces), but the algorithm computed `bullish` (confidence 0.45) off a bearish-to-bullish CHoCH. This is a legitimate disagreement rather than a bug - "neutral" during chop is a judgment call the algorithm's rules don't model, since it only ever reports the direction of the latest BOS/CHoCH, with no explicit "no clear structure" state.

## CI Integration
`.github/workflows/eval-trading-research-copilot.yml` runs this harness automatically on any PR touching `01-trading-research-copilot/**` and posts both rates as a PR comment. Report-only - it does not gate the PR. Since the harness makes no LLM/API calls, the workflow requires no secrets to run this step (secrets are only needed elsewhere in this project, for `run_copilot.py`/`demo.py`'s LLM synthesis and LangSmith tracing).

## Limitations
- **This is a research/synthesis tool, not a live execution system.** It operates on delayed `yfinance` data with no order routing, position sizing, or execution capability of any kind - see the project README. Nothing in this eval, including a high bias-accuracy score, should be read as validating the system for live trading decisions.
- **`expected_bias` is a human's visual judgment call, not an objective ground truth.** There is no independently verifiable "correct" bias for a given chart - especially for choppy periods, where a reasonable reviewer could defensibly call the same chart `neutral` or lean `bullish`/`bearish`. Read "Bias Accuracy" as **agreement-with-a-human-reviewer**, not correctness in an absolute sense. A low score could mean the algorithm is wrong, or that the label itself was a closer call than it looked.
- **The candidate sanity check does not use point-in-time LTF data.** `yfinance`'s 5-minute intraday data only reaches back ~60 days from *today*, not 60 days from an arbitrary historical `end_date` - so for golden-dataset dates that are themselves weeks old, a temporally matched LTF window often isn't available at all. The harness instead runs `find_entry_candidates()` against *current* live LTF data alongside each entry's *historical* HTF bias. This exercises the candidate-ranking logic structurally (are zones well-formed and correctly tiered) but is **not** a backtest of what the LTF candidates would actually have looked like on that historical date.
- **5 entries is a small sample.** It's enough to catch a clearly broken bias rule or a ranking bug, not enough to estimate a statistically reliable accuracy rate. Expanding the golden dataset (more dates, more instruments, more choppy examples specifically) would tighten this.

## Design Notes
- `compute_htf_bias()` and `find_entry_candidates()` are both pure/deterministic given their input DataFrame, which is what makes this eval possible without any LLM judge or mocking - the same property that lets `graph/build_graph.py`'s pipeline defer LLM use to only the final synthesis node.
- The golden dataset intentionally includes one `neutral` (choppy) label, since a bias engine that only ever predicts `bullish`/`bearish` would look deceptively accurate on trending-only test data.
