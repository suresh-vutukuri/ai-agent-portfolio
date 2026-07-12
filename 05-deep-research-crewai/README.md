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

**Evaluation:** 5 research topics, checking citation quality across ~20 citations per report (100 citations total) on three dimensions: is the URL well-formed and reachable, does it match a real search result (no fabrication), and does an LLM-judge (gpt-4o-mini) confirm the cited claim is actually supported by that source's content.

| Metric | Score |
|---|---|
| URL Well-Formed | 100% |
| URL Reachable (direct) | 88% |
| URL Bot-Blocked (link is live, checker denied) | 12% |
| URL Dead (genuinely broken) | 0% |
| Source Matched (no fabricated citations) | 98% |
| Avg. Groundedness (LLM-judge, 1-5) | 3.75 |
| Citation Validity Score (composite, bot-blocks excluded) | 49% |

*Bot-blocked ≠ dead: academic/publisher sites (MDPI, ScienceDirect, IEEE, etc.) sit behind WAFs that reject scripted HTTP requests via TLS fingerprinting and JS challenges, regardless of headers — confirmed by manually reproducing the block outside the eval. These are tracked separately and excluded from penalizing the validity score, since the underlying links are live (see `docs/eval_harness.md`).*

*The composite validity score is held down mainly by groundedness, not fabrication or dead links — most citations are real and traceable (98% source-matched), but a meaningful share of claims are only loosely supported by their cited source rather than tightly grounded. A writer-prompt change (one citation per specific claim, no combining facts across sources) was tested and did not measurably improve this — documented as a tested-but-inconclusive iteration rather than a fix, in `docs/eval_harness.md`.*

**Why it matters:** Demonstrates the planning → search → synthesis agent pattern with citation grounding treated as a measured, adversarially-tested property rather than an assumed one — including transparently reporting where an eval fix didn't work, and correctly diagnosing a false-negative source (bot-blocking) instead of misattributing it to broken links.

**Stack:** CrewAI (sequential process) · Tavily Search · GPT-4o-mini · Custom citation-validity eval harness
