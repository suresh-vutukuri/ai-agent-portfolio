## HR Policy RAG Agent (Pinecone)

**Problem:** Employees need fast, accurate answers to HR policy questions (PTO, benefits, leave, remote work, 401k) without digging through scattered docx/PDF documents.

**Approach:** RAG pipeline with format-aware ingestion (docx + PDF), token-based chunking, OpenAI embeddings, and Pinecone as the cloud vector store. An LCEL chain retrieves top-4 relevant chunks and generates cited answers, refusing to answer when policy coverage is insufficient — a hallucination guardrail for HR/compliance contexts.

**Architecture**
`[User query] → Retriever (Pinecone, top-k=4) → LCEL Chain (prompt | gpt-4o-mini | parser) → Cited answer`

**Folder Structure**
- `sample_docs/` — 5 dummy HR policy docs (mixed .docx/.pdf)
- `ingestion/` — document loaders + token-based chunker
- `retrieval/` — embedding + Pinecone index connection, retriever
- `chain/` — LCEL RAG chain with citation formatting
- `pipeline/` — build_index.py (run once to embed + upsert)
- `eval/` — golden Q&A set + custom eval harness (recall@k, groundedness, citation accuracy, relevance)

**Setup**

> **Note:** Requires Python 3.12 — `langchain-pinecone` is not compatible with Python 3.14.

\`\`\`bash
pip install -r requirements.txt
cp .env.example .env  # add OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME
python pipeline/build_index.py
python chain/query.py "How many PTO days do I get after 4 years?"
\`\`\`

**Evaluation:** Same custom eval harness and golden dataset (25 questions: 22 answerable + 3 negative controls) used across both vector store versions, for a like-for-like comparison.

| Metric | Score |
|---|---|
| Recall@4 | 100% |
| Groundedness | 4.82 / 5 |
| Citation Accuracy | 95% |
| Answer Relevance | 4.73 / 5 |
| Negative Control Refusal Accuracy | 100% |

*Scores are identical to the Chroma version — expected, since retrieval quality here is driven by chunking/embedding strategy, not vector store choice, at this corpus size. Value of Pinecone in this setup is operational (cloud-hosted, scalable), not accuracy.*

**Why it matters:** Demonstrates the same eval-driven RAG pipeline deployed against a production-grade managed vector database, with a documented understanding of when a cloud vector store's value shows up (scale, ops) versus when it doesn't (small corpus retrieval quality).

**Vector store note:** This is the Pinecone (cloud) version. See `02a-hr-policy-rag-chroma/` for the Chroma (local) version and cost/latency trade-off comparison.

**Stack:** LangChain (LCEL) · Pinecone · OpenAI Embeddings · GPT-4o-mini · Custom Python eval harness