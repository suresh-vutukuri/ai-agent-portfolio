# Eval Harness Methodology — HR Policy RAG (Pinecone)

## Purpose
This document describes the custom evaluation harness used to measure the quality of the HR Policy RAG agent's retrieval and generation, running against a Pinecone (cloud) vector store. The harness, golden dataset, and metrics are identical to the Chroma version (`02a-hr-policy-rag-chroma/`) to enable a like-for-like comparison — only the underlying vector store differs.

## Golden Dataset
- **Location:** `eval/golden_qa_eval_set.json`
- **Size:** 25 question-answer pairs
  - 22 answerable questions, each mapped to an expected `source_doc` and `source_section`
  - 3 negative controls — questions with no answer in the policy documents, used to test refusal behavior instead of hallucination
- **Categories:** `factual_lookup`, `eligibility_reasoning`, `calculation`, `negative_control`

## Metrics

### 1. Recall@k (k=4)
Checks whether the retriever's top-4 retrieved chunks (from the Pinecone index) include the expected `source_doc` for a given question. Binary per question, averaged across all answerable questions.

**What it measures:** Retrieval quality independent of generation — is the right document being surfaced at all.

### 2. Groundedness
An LLM-judge (gpt-4o-mini) scores 1–5 whether the generated answer is fully supported by the retrieved context, penalizing any claims not present in the source chunks.

**What it measures:** Hallucination risk — does the model stick to what's actually in the documents.

### 3. Citation Accuracy
Checks whether the source cited in the generated answer matches the expected `source_doc` for that question. Binary per question, averaged.

**What it measures:** Whether the agent is citing the correct document, not just any document.

### 4. Answer Relevance
An LLM-judge (gpt-4o-mini) scores 1–5 how well the generated answer addresses the question compared to the `expected_answer`.

**What it measures:** Overall answer quality/usefulness, independent of grounding.

### 5. Negative Control Refusal Accuracy
For the 3 unanswerable questions, checks whether the agent correctly refuses (e.g., "Not covered in the available policy documents") rather than fabricating an answer. Binary per question, averaged.

**What it measures:** Whether the guardrail against answering outside scope actually works.

## Running the Harness
```bash
python eval/run_eval.py
```
This runs all 25 questions through `chain/rag_chain.py` (backed by the Pinecone retriever), computes the metrics above, and writes:
- `eval/results/scorecard.csv` — per-question scores
- `eval/results/summary.json` — aggregate averages
- `eval/results/scorecard.png` — bar chart (via `eval/plot_results.py`)

Requires `PINECONE_API_KEY` and `PINECONE_INDEX_NAME` set in `.env`, and the index already populated via `pipeline/build_index.py`.

## Latest Results
| Metric | Score |
|---|---|
| Recall@4 | 100% |
| Groundedness | 4.82 / 5 |
| Citation Accuracy | 95% |
| Answer Relevance | 4.73 / 5 |
| Negative Control Refusal Accuracy | 100% |

Scores match the Chroma version exactly — at this corpus size, retrieval quality is driven by chunking/embedding strategy rather than vector store choice. See `docs/vector_store_comparison.md` for where Pinecone's value actually shows up (scale, ops, managed infrastructure) versus where it doesn't (small-corpus accuracy).

## CI Integration
`.github/workflows/eval-hr-rag-pinecone.yml` runs this harness automatically on any PR touching `02b-hr-policy-rag-pinecone/`, posting the resulting scores as a PR comment. Currently report-only (no merge gate).

## Design Notes
- Metrics are intentionally lightweight (LLM-judge + rule-based checks) rather than a heavyweight eval framework, keeping the harness transparent — every score traces back to a specific check in `eval/metrics.py`.
- Negative controls are scored separately from answerable questions since a low groundedness/relevance score would be meaningless for a question with no valid answer.
- Harness code is shared verbatim with `02a-hr-policy-rag-chroma/eval/` — only the retriever backing the chain differs between versions.
