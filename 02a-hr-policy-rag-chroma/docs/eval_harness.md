# Eval Harness Methodology — HR Policy RAG (Chroma)

## Purpose
This document describes the custom evaluation harness used to measure the quality of the HR Policy RAG agent's retrieval and generation, and how to run it.

## Golden Dataset
- **Location:** `eval/golden_qa_eval_set.json`
- **Size:** 25 question-answer pairs
  - 22 answerable questions, each mapped to an expected `source_doc` and `source_section`
  - 3 negative controls — questions with no answer in the policy documents, used to test refusal behavior instead of hallucination
- **Categories:** `factual_lookup`, `eligibility_reasoning`, `calculation`, `negative_control`

## Metrics

### 1. Recall@k (k=4)
Checks whether the retriever's top-4 retrieved chunks include the expected `source_doc` for a given question. Binary per question (1 = found, 0 = not found), averaged across all answerable questions.

**What it measures:** Retrieval quality independent of generation — is the right document being surfaced at all.

### 2. Groundedness
An LLM-judge (gpt-4o-mini) scores 1–5 whether the generated answer is fully supported by the retrieved context, penalizing any claims not present in the source chunks.

**What it measures:** Hallucination risk — does the model stick to what's actually in the documents.

### 3. Citation Accuracy
Checks whether the source cited in the generated answer (e.g., `[Source: filename, Section X]`) matches the expected `source_doc` for that question. Binary per question, averaged.

**What it measures:** Whether the agent is citing the correct document, not just any document.

### 4. Answer Relevance
An LLM-judge (gpt-4o-mini) scores 1–5 how well the generated answer addresses the question compared to the `expected_answer`.

**What it measures:** Overall answer quality/usefulness, independent of grounding.

### 5. Negative Control Refusal Accuracy
For the 3 unanswerable questions, checks whether the agent correctly responds with a refusal (e.g., "Not covered in the available policy documents") rather than fabricating an answer. Binary per question, averaged.

**What it measures:** Whether the guardrail against answering outside scope actually works.

## Running the Harness
```bash
python eval/run_eval.py
```
This runs all 25 questions through `chain/rag_chain.py`, computes the metrics above, and writes:
- `eval/results/scorecard.csv` — per-question scores
- `eval/results/summary.json` — aggregate averages
- `eval/results/scorecard.png` — bar chart (via `eval/plot_results.py`)

## Latest Results
| Metric | Score |
|---|---|
| Recall@4 | 100% |
| Groundedness | 4.82 / 5 |
| Citation Accuracy | 95% |
| Answer Relevance | 4.73 / 5 |
| Negative Control Refusal Accuracy | 100% |

## CI Integration
`.github/workflows/eval-hr-rag-chroma.yml` runs this harness automatically on any PR touching `02a-hr-policy-rag-chroma/`, posting the resulting scores as a PR comment. Currently report-only (no merge gate); a score threshold gate may be added later once a baseline stability window is established.

## Design Notes
- Metrics are intentionally lightweight (LLM-judge + rule-based checks) rather than a heavyweight eval framework, to keep the harness transparent and easy to reason about — every score can be traced back to a specific check in `eval/metrics.py`.
- Negative controls are scored separately from answerable questions since a low score on groundedness/relevance would be meaningless for a question with no valid answer.
