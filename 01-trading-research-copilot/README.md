## Trading Research Copilot (LangGraph + LangSmith)

**Important disclaimer:** This is a research/synthesis tool illustrating a multi-timeframe agent architecture on ES=F/NQ=F. It is explicitly **not a real-time execution system**, and any suggested entries are historical-context research reads based on delayed yfinance data — not live tradeable signals.

**Problem:** Reading multi-timeframe market structure (HTF bias, LTF entry zones) manually is time-consuming and easy to do inconsistently. This agent automates the ICT/SMC-style read — HTF bias, then LTF entry candidates filtered by that bias — into a synthesized research note.

**Approach:** A LangGraph pipeline fetches 1-hour bars to establish HTF bias (via swing structure, BOS/CHoCH, FVG/order block detection), fetches 5-minute bars to find LTF entry candidates (FVGs/OBs aligned with HTF bias, with liquidity-sweep and distance-from-price weighting), and an LLM node synthesizes the results into a structured markdown note — separating near-term candidates from standing reference zones.

**Architecture**
[Ticker] → Fetch HTF (1H) → Compute HTF Bias (swings, BOS/CHoCH, FVG/OB)
→ Fetch LTF (5min) → Find LTF Candidates (bias-filtered, sweep + distance ranked)
→ Synthesize (LLM) → Research note (near-term vs. standing reference zones)

**Folder Structure**
- `data/` — yfinance fetch layer for HTF (1H) and LTF (5min) bars
- `analysis/` — swing structure, BOS/CHoCH, FVG/order block detection, liquidity sweeps, LTF candidate ranking
- `graph/` — LangGraph state, nodes, synthesis prompt, graph assembly
- `eval/` — golden dataset (manually labeled bias) + eval harness
- `run_copilot.py` — CLI entry point
- `demo.py` — runs both ES=F and NQ=F with today's data, zero setup beyond `.env`

**Setup**
```bash
pip install -r requirements.txt
cp .env.example .env  # add OPENAI_API_KEY, LANGCHAIN_API_KEY (for LangSmith tracing)
python run_copilot.py --ticker ES=F
python demo.py
```

**Evaluation:** 5 historical dates (3 ES=F, 2 NQ=F) with manually chart-reviewed expected HTF bias, checking (1) bias-accuracy against that human label and (2) sanity of the top near-term candidate (valid invalidation level, genuinely within the near-term distance threshold).

| Metric | Score |
|---|---|
| Bias Accuracy | 80% (4/5) |
| Candidate Sanity Check | 100% (5/5) |

*The one mismatch: ES=F on 2026-06-26, a choppy/neutral period a human reviewer labeled neutral, which the algorithm read as bullish via a CHoCH signal — a reasonable rules-based read of a genuinely ambiguous, low-conviction structure shift, not a clear detection error. See `docs/eval_harness.md` for the full methodology and a note that "expected bias" is a human judgment call, not an objective ground truth — this metric measures agreement with a reviewer, not correctness in an absolute sense.*

**Why it matters:** Demonstrates a multi-node LangGraph pipeline (fetch → rule-based detection → LLM synthesis) with LangSmith tracing for observability, and an eval harness grounded in a domain the author has deep, verifiable expertise in — including catching and fixing a real ranking bug (distant unmitigated zones outranking near-term ones) found through manual chart verification during development, not assumed to be correct.

**Stack:** LangGraph · LangSmith · yfinance · GPT-4o-mini · Custom bias-accuracy + candidate-sanity eval harness