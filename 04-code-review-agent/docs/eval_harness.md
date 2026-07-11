# Eval Harness Methodology — Code Review / PR Agent

## Purpose
This document describes the evaluation harness used to measure the accuracy of the code review agent's findings against a set of synthetic diffs with known, labeled issues.

## Test Dataset
- **Location:** `eval/test_diffs/`
- **Size:** 5 synthetic Python diffs, each with 1-2 deliberately injected issues spanning:
  - Security (e.g., SQL injection risk, hardcoded secret)
  - Style/convention (e.g., unused import, bare `except`)
  - Logic (e.g., off-by-one error)
- **Labels:** `eval/test_diffs/expected_issues.json` — file, line, and issue type for each injected issue.

## Metrics

### Precision (full-scope and critical-scoped)
Of all issues the agent flagged, what fraction correspond to a real labeled issue.
- **Full-scope (`micro_precision`/`macro_precision`):** counts all flagged issues across every severity section. Tends to look low because the review rubric intentionally also flags legitimate issues (missing test coverage, style nits) beyond the narrow hand-picked `expected_issues.json` set — these are real findings, not hallucinations, just outside the eval's labeled scope.
- **Critical-scoped (`critical_precision`):** counts only flagged issues under the agent's `## Critical` severity section, where injected bugs are expected to land. This is the more meaningful signal for "did the agent hallucinate on the things being tested."

### Recall (full-scope and critical-scoped)
Of all labeled issues actually present, what fraction the agent's review caught.
- **Full-scope (`micro_recall`/`macro_recall`):** counts a match anywhere in the review output, any severity section.
- **Critical-scoped (`critical_recall`):** counts a match only if it landed in the `## Critical` section — stricter, penalizes correct catches that got severity-misclassified.

**Matching logic:** An agent-flagged issue counts as a match if it references the same file and line (±1 line tolerance) and a semantically equivalent issue type to a labeled issue.

## Model Comparison
The harness was run against three Claude models to select the right cost/accuracy tradeoff for a PR-triggered agent (potentially running on every PR):

| Metric | Haiku | Sonnet (default) | Opus |
|---|---|---|---|
| Recall (all flagged issues) | 14% | 86% | 100% |
| Critical Recall | 14% | 71% | 71% |
| Critical Precision | 10% | 83% | 83% |
| True Positives | 1/7 | 6/7 | 7/7 |

**Decision: Sonnet.** Opus's only edge is 1 additional true positive on full-scope recall; critical-section performance (the stricter, more meaningful metric) is identical to Sonnet. Given the cost delta and that this agent may run frequently in CI, Sonnet is the better default. Haiku's reasoning is too weak for reliable logic/security bug detection — not recommended for this task.

## Running the Harness
```bash
python eval/run_eval.py
```
Runs each test diff through `agent/reviewer.py` (using `ANTHROPIC_MODEL` from `.env`, default Sonnet), compares flagged issues against `expected_issues.json`, and writes:
- `eval/results/scorecard.csv` — per-diff results (expected vs. flagged issues)
- `eval/results/summary.json` — aggregate precision/recall (full-scope and critical-scoped)

## Latest Results (Sonnet, default model)
| Metric | Score |
|---|---|
| Critical Precision | 83% |
| Critical Recall | 71% |
| Recall (all flagged issues) | 86% |

## CI Integration
This harness is currently run manually/locally rather than gated in CI, since the review agent's separate `.github/workflows/code-review-on-pr.yml` action is scoped to live PR review, not the synthetic eval suite. The eval suite can be added as a separate report-only workflow later, following the same pattern as the other agents.

## Design Notes
- Static analysis findings (ruff, bandit) are deterministic and not separately scored — the eval focuses on the agent's overall review output, which combines linter findings with LLM judgment on logic/design issues that linters can't catch.
- Small, deliberately simple synthetic diffs were used (rather than real historical PRs) to keep expected issues unambiguous and avoid subjective grading disputes.
