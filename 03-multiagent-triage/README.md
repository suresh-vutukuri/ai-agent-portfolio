## Multi-Agent Triage Router (CrewAI)

**Problem:** Customer support queries need to be routed to the right specialist quickly and accurately — billing, technical, or returns — without a human triaging every ticket manually.

**Approach:** A hierarchical CrewAI crew with a Triage Manager agent that classifies each incoming query and delegates to one of three specialist agents (Billing, Technical Support, Returns), each equipped with mock tools (invoice lookup, system status, order/refund lookup) backed by synthetic data. The manager reviews the specialist's response before finalizing it.

**Architecture**
`[Customer query] → Triage Manager (classify + delegate) → Specialist Agent (billing/tech/returns, uses tools) → Manager review → Final response`

**Folder Structure**
- `agents/config/agents.yaml` — role, goal, backstory for manager + 3 specialists
- `agents/config/tasks.yaml` — triage and resolve task definitions
- `agents/crew.py` — `@CrewBase` crew assembly (hierarchical process)
- `agents/run_triage.py` — `run_triage(query)` entry point used by `demo.py` and the eval harness
- `agents/handoff_logger.py` — logs each run's routing/tool-call trace to `logs/handoff_log.jsonl`
- `tools/` — mock billing, tech, and returns tools (synthetic in-memory data)
- `eval/` — labeled test queries + routing-accuracy harness
- `demo.py` — runnable demo with hardcoded sample queries (no setup beyond `.env`)

**Setup**
```bash
pip install -r requirements.txt
cp .env.example .env  # add OPENAI_API_KEY
python demo.py
```
Requires Python 3.9–3.13 — `crewai` pulls in `chromadb`, whose pydantic v1 compat shim breaks on 3.14 (see `requirements.txt`).

**Evaluation:** 15 labeled customer queries (5 per specialist category), checking whether the crew routed each to the correct specialist.

| Metric | Score |
|---|---|
| Routing Accuracy | 93.3% (14/15) |

*See `eval/results/routing_scorecard.csv` for the per-query breakdown, including the one misrouted case.*

**Why it matters:** Demonstrates the manager/delegate orchestration pattern — a common real-world agent architecture — with observability into every handoff (logged tool calls, routing decisions) rather than a black-box crew.

**Stack:** CrewAI (hierarchical process) · OpenAI (`gpt-4.1-mini`, CrewAI's default — not pinned in `agents.yaml`) · YAML agent config · Custom routing-accuracy eval