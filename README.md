# AI Agent Portfolio

A collection of AI agents built to explore agent orchestration patterns, retrieval-augmented generation, and — most importantly — **eval-driven development**: every agent here ships with a custom evaluation harness, not just a working demo. Where relevant, agents are also compared across models and infrastructure choices (vector stores, LLMs) with results reported honestly, including where a change didn't help.

## Agents

| # | Agent | Framework | Purpose | Headline Eval Result |
|---|---|---|---|---|
| 01 | [Trading Research Copilot](./01-trading-research-copilot) | LangGraph + LangSmith | Multi-timeframe (1H bias, 5-min entry) market structure research tool for ES=F/NQ=F | 80% bias accuracy vs. human review, 100% candidate sanity |
| 02a | [HR Policy RAG — Chroma](./02a-hr-policy-rag-chroma) | LangChain (LCEL) | Q&A over HR policy docs, local vector store | 100% recall@4, 95% citation accuracy |
| 02b | [HR Policy RAG — Pinecone](./02b-hr-policy-rag-pinecone) | LangChain (LCEL) | Same agent, cloud vector store — see [comparison](./docs/vector_store_comparison.md) | Identical accuracy to 02a; differs in cost/ops tradeoffs |
| 03 | [Multi-Agent Triage Router](./03-multiagent-triage) | CrewAI (hierarchical) | Routes customer support queries to billing/tech/returns specialists | 93.3% routing accuracy |
| 04 | [Code Review / PR Agent](./04-code-review-agent) | Claude Agent SDK | Reviews PR diffs (ruff + bandit + LLM reasoning), posts structured comments | 83% critical-issue precision (Sonnet) |
| 05 | [Deep Research Agent](./05-deep-research-crewai) | CrewAI (sequential) | Plans, searches (Tavily), and synthesizes a cited research report on a topic | 100% source-matched, 4.04/5 avg. groundedness |


## What This Portfolio Demonstrates

- **Multiple orchestration patterns:** graph-based state machines (LangGraph), role-based crews (CrewAI, hierarchical and sequential), and native tool-use agents (Claude Agent SDK)
- **Eval-driven development, not just demos:** every agent has a custom eval harness with a defined metric, a golden/test dataset, and honestly reported results — including negative findings (e.g., a prompt change tested on the Deep Research agent that didn't improve citation validity, reported as-is rather than hidden)
- **Infrastructure tradeoff analysis:** the HR RAG agent is built twice (Chroma vs. Pinecone) specifically to compare and document where a "bigger" infrastructure choice does and doesn't matter
- **Model selection based on evidence:** the Code Review agent tests Haiku vs. Sonnet vs. Opus and documents why Sonnet was chosen over the more expensive option
- **CI-integrated evals:** each agent has a GitHub Actions workflow that runs its eval harness on every PR and posts results as a comment (report-only, not a merge gate)
- **Domain-grounded work, not generic demos:** the Trading Research Copilot is built on real ICT/SMC market structure methodology, with output manually verified against live charts during development (catching and fixing a real ranking bug in the process — documented in that agent's build history)

## Repo Structure
```
ai-agent-portfolio/
├── 01-trading-research-copilot/
├── 02a-hr-policy-rag-chroma/
├── 02b-hr-policy-rag-pinecone/
├── 03-multiagent-triage/
├── 04-code-review-agent/
├── 05-deep-research-crewai/
├── docs/
│   └── vector_store_comparison.md
└── .github/workflows/          # per-agent eval-on-PR workflows
```
Each agent folder is self-contained: its own `README.md`, `requirements.txt`, `docs/eval_harness.md`, and `eval/` results.

## Tech Stack
LangChain · LangGraph · LangSmith · CrewAI · Claude Agent SDK · OpenAI (GPT-4o-mini) · Anthropic (Claude) · Chroma · Pinecone · Tavily · yfinance · Python · GitHub Actions

## Running Any Agent
Each agent has its own setup instructions in its README, but the general pattern is:
```bash
cd <agent-folder>
pip install -r requirements.txt
cp .env.example .env   # add the required API keys for that agent
python <entry-point>.py
```

## Author
Built by [Suresh Vutukuri](https://github.com/suresh-vutukuri) as a hands-on exploration of production AI agent patterns — orchestration, retrieval, evaluation, and observability.
