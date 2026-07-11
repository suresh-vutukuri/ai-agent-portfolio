## Code Review / PR Agent (Claude Agent SDK)

**Problem:** Manual PR review is slow and inconsistent — security issues, style violations, and logic bugs can slip through, especially on smaller teams without dedicated review bandwidth.

**Approach:** An agent built on the Claude Agent SDK that fetches a PR's diff, runs static analysis (`ruff` for style/lint, `bandit` for security), and combines those findings with an LLM-driven review of logic and test coverage — producing a single structured, severity-tagged review comment.

**Architecture**
`[PR opened/updated] → Fetch diff (PyGithub) → Parse diff → Run ruff + bandit → Claude review (diff + linter findings) → Structured markdown review → Post as PR comment`

**Folder Structure**
- `tools/diff_parser.py` — unified diff parsing
- `tools/lint_tools.py` — ruff + bandit wrappers
- `tools/github_client.py` — PR fetch + comment posting (PyGithub)
- `agent/review_prompt.py` — review rubric (security, style, logic, test coverage)
- `agent/reviewer.py` — main review loop (Claude Agent SDK)
- `agent/run_review.py` — CLI entry point (`--post` flag required to actually post to GitHub)
- `eval/` — synthetic test diffs with labeled issues + precision/recall harness
- `sandbox/` — sample files for triggering test PRs (isolated from the agent's own code)
- `action.yml` — reusable GitHub composite action

**Setup**

> **Prerequisite:** Requires a GitHub Personal Access Token (fine-grained, **repo read + pull request write** scope, **only select repositories**). Generate your own token — do not use or share someone else's — and store it as `GITHUB_TOKEN` in `.env`. Recommend a longer expiration (e.g., 90 days) if you plan to demo this over time.

```bash
pip install -r requirements.txt
cp .env.example .env  # add ANTHROPIC_API_KEY, GITHUB_TOKEN
python agent/run_review.py --repo owner/repo --pr 12          # print review only
python agent/run_review.py --repo owner/repo --pr 12 --post   # print AND post to GitHub
```

**Testing locally:** `.github/workflows/code-review-on-pr.yml` triggers the agent automatically on PRs touching `sandbox/` only — scoped to avoid burning tokens on every portfolio PR. Edit a file in `sandbox/` and open a PR to see it in action.

**Using this as a reusable action in another repo:**
```yaml
# .github/workflows/ai-review.yml in the other repo
name: AI Code Review
on:
  pull_request:
    types: [opened, synchronize]

jobs:
  review:
    runs-on: ubuntu-latest
    permissions:
      pull-requests: write
      contents: read
    steps:
      - uses: suresh-vutukuri/ai-agent-portfolio/04-code-review-agent@main
        with:
          github-token: ${{ secrets.GITHUB_TOKEN }}
          anthropic-api-key: ${{ secrets.ANTHROPIC_API_KEY }}
```
Requires the portfolio repo to be public, and the consuming repo to supply its own `ANTHROPIC_API_KEY` secret (`GITHUB_TOKEN` is auto-provided by GitHub Actions). For stability, reference a tagged release (e.g., `@v1.0.0`) instead of `@main` once tagged.

**Evaluation:** 5 synthetic diffs with 7 deliberately injected issues (security risks, style violations, logic bugs), checking whether the agent's review correctly flags each labeled issue. Tested across three Claude models to select the right cost/accuracy tradeoff for a PR-triggered agent.

| Metric | Haiku | Sonnet (default) | Opus |
|---|---|---|---|
| Recall (all flagged issues) | 14% | 86% | 100% |
| Critical Recall | 14% | 71% | 71% |
| Critical Precision | 10% | 83% | 83% |
| True Positives | 1/7 | 6/7 | 7/7 |

*Critical precision/recall scope the match to issues the agent classified under its `## Critical` severity section — the metric that best reflects "did it hallucinate on the things being tested" (full-scope precision looks artificially low because the review rubric intentionally also flags legitimate style/test-coverage issues outside the labeled set).*

**Model choice: Sonnet.** Opus edges out Sonnet on total recall but performs identically on critical-section accuracy, at meaningfully higher cost — not worth it for an agent that may run on every PR. Haiku is not recommended for this task; its reasoning is too weak to reliably catch injected logic/security issues. `ANTHROPIC_MODEL` defaults to Sonnet in `.env.example`.

**Why it matters:** Directly relatable to engineering hiring managers — automates a task every dev team does, combining deterministic tooling (linters) with LLM judgment (logic/design review), and ships as a reusable GitHub Action rather than a standalone script.

**Stack:** Claude Agent SDK · PyGithub · ruff · bandit · Custom precision/recall eval harness
