## HR Policy RAG Agent (Chroma)

**Problem:** Employees need fast, accurate answers to HR policy questions (PTO, benefits, leave, remote work, 401k) without digging through scattered docx/PDF documents.

**Approach:** RAG pipeline with format-aware ingestion (docx + PDF), token-based chunking, OpenAI embeddings, and Chroma as the vector store. An LCEL chain retrieves top-4 relevant chunks and generates cited answers, refusing to answer when policy coverage is insufficient — a hallucination guardrail for HR/compliance contexts.

**Architecture**
`[User query] → Retriever (Chroma, top-k=4) → LCEL Chain (prompt | gpt-4o-mini | parser) → Cited answer`

**Folder Structure**
- `sample_docs/` — 5 dummy HR policy docs (mixed .docx/.pdf)
- `ingestion/` — document loaders + token-based chunker
- `retrieval/` — embedding + Chroma store, retriever
- `chain/` — LCEL RAG chain with citation formatting
- `pipeline/` — build_index.py (run once to embed + persist)
- `eval/` — golden Q&A set + custom eval harness (recall@k, groundedness, citation accuracy, relevance)
- `db/` — persisted Chroma DB (gitignored, regenerate via pipeline)

**Setup**
\`\`\`bash
pip install -r requirements.txt
cp .env.example .env  # add OPENAI_API_KEY
python pipeline/build_index.py
python chain/query.py "How many PTO days do I get after 4 years?"
\`\`\`

**Evaluation:** Custom eval harness against a 25-question golden dataset (22 answerable + 3 negative controls), measuring retrieval recall, groundedness, citation accuracy, and answer relevance.

| Metric | Score |
|---|---|
| Recall@4 | 100% |
| Groundedness | 4.82 / 5 |
| Citation Accuracy | 95% |
| Answer Relevance | 4.73 / 5 |
| Negative Control Refusal Accuracy | 100% |

*See `eval/results/scorecard.csv` for per-question breakdown and `eval/results/scorecard.png` for the chart.*

**Why it matters:** Built with production discipline — measurable retrieval quality, hallucination guardrails, and a reproducible eval pipeline that catches regressions before they ship.

**Vector store note:** This is the Chroma (local) version. See `02b-hr-policy-rag-pinecone/` for the Pinecone (cloud) version, with a documented cost/latency trade-off comparison.

**Stack:** LangChain (LCEL) · Chroma · OpenAI Embeddings · GPT-4o-mini · Custom Python eval harness