# Eval Harness Methodology — Multi-Agent Triage Router

## Purpose
This document describes the evaluation harness used to measure routing accuracy of the CrewAI triage crew — i.e., whether customer queries are delegated to the correct specialist agent.

## Test Dataset
- **Location:** `eval/test_queries.json`
- **Size:** 15 labeled customer queries
  - 5 billing-related queries
  - 5 technical support queries
  - 5 returns-related queries
- Each entry has a `query` (realistic customer phrasing) and an `expected_specialist` (`billing`, `tech`, or `returns`) label.

## Metric

### Routing Accuracy
For each query, the harness runs the full crew via `agents/run_triage.py`, captures which specialist agent was actually invoked (via CrewAI's task execution trace / handoff log), and compares it against `expected_specialist`. Binary per query (1 = correct specialist invoked, 0 = incorrect), averaged across all 15 queries.

**What it measures:** Whether the Triage Manager's classification and delegation logic is reliably routing queries to the right domain expert — the core function of a hierarchical multi-agent system.

## Running the Harness
```bash
python eval/run_eval.py
```
This runs all 15 queries through the crew, logs each routing decision, and writes:
- `eval/results/routing_scorecard.csv` — per-query result (query, expected specialist, actual specialist, correct/incorrect)
- `eval/results/summary.json` — aggregate routing accuracy

## Latest Results
| Metric | Score |
|---|---|
| Routing Accuracy | 93.3% (14/15) |

## CI Integration
`.github/workflows/eval-multiagent-triage.yml` runs this harness automatically on any PR touching `03-multiagent-triage/`, posting routing accuracy as a PR comment. Currently report-only (no merge gate).

## Design Notes
- Routing accuracy is intentionally the only automated metric here — response quality from each specialist is a secondary concern to the crew's core job (correct delegation), and is instead spot-checked via `demo.py` transcripts rather than a scored metric.
- Handoff logs (`logs/handoff_log.jsonl`) provide a full trace of tool calls and agent decisions per run, useful for manually inspecting the one misrouted case in the scorecard.
