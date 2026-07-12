## Deep Research Agent (CrewAI)

**Problem:** Researching a topic thoroughly requires breaking it into sub-questions, searching multiple sources, and synthesizing findings with accurate citations — a slow manual process prone to shallow or unsourced summaries.

**Approach:** A sequential CrewAI crew: a Planner agent breaks the topic into sub-questions, a Researcher agent searches the web (Tavily) to answer each one, and a Writer agent synthesizes findings into a cited markdown report. Citation quality is measured with a custom eval harness rather than assumed.

**Architecture**
`[Topic] → Planner (sub-questions) → Researcher (Tavily search per sub-question) → Writer (cited synthesis) → Markdown report`

**Folder Structure**
- `agents/config/agents.yaml` — role, goal, backstory for planner, researcher, writer
- `agents/config/tasks.yaml` — plan → research → synthesize task chain
- `agents/crew.py` — `@CrewBase` crew assembly (sequential process)
- `tools/search_tools.py` — Tavily search wrapper
- `tools/citation_tracker.py` — claim-to-source mapping
- `eval/` — test topics + citation-validity eval harness
- `sandbox/` — demo/testing files
- `run_research.py` — CLI entry point
- `demo.py` — runnable demo with hardcoded example topics

**Setup**
```bash
pip install -r requirements.txt
cp .env.example .env  # add OPENAI_API_KEY, TAVILY_API_KEY
python run_research.py --topic "Is ICT's trading model profitable for beginners?"
```

**Evaluation:** 5 research topics, checking citation quality across ~22 citations per report on three dimensions: is the URL well-formed and reachable, does it match a real search result (no fabrication), and does an LLM-judge (gpt-4o-mini) confirm the cited claim is actually supported by that source's content. Latest scores from the CI eval-on-PR run:

| Metric | Score |
|---|---|
| URL Well-Formed | 100% |
| URL Reachable (direct) | 86.6% |
| URL Bot-Blocked (link is live, checker denied) | 10.7% |
| URL Dead (genuinely broken) | 2.7% |
| Source Matched (no fabricated citations) | 100% |
| Avg. Groundedness (LLM-judge, 1-5) | 4.04 |
| Citation Validity Score (composite, bot-blocks excluded) | 58.9% |

*Bot-blocked ≠ dead: academic/publisher sites (MDPI, ScienceDirect, IEEE, etc.) sit behind WAFs that reject scripted HTTP requests via TLS fingerprinting and JS challenges, regardless of headers — confirmed by manually reproducing the block outside the eval. These are tracked separately and excluded from penalizing the validity score, since the underlying links are live (see `docs/eval_harness.md`).*

*Scores vary somewhat run-to-run since fresh web searches return different results each time — an earlier local run scored 48.7% validity with a 3.75 judge average; this CI run scored higher. Both are documented in `docs/eval_harness.md`, including a writer-prompt change that was tested and found not to reliably move the score in either direction — a transparently-reported inconclusive iteration rather than a claimed fix.*

**Why it matters:** Demonstrates the planning → search → synthesis agent pattern with citation grounding treated as a measured, adversarially-tested property rather than an assumed one — including transparently reporting run-to-run variance and a tested-but-inconclusive prompt iteration, and correctly diagnosing a false-negative source (bot-blocking) instead of misattributing it to broken links.

**Stack:** CrewAI (sequential process) · Tavily Search · GPT-4o-mini · Custom citation-validity eval harness