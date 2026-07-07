# Vector Store Comparison: Chroma vs. Pinecone

## Purpose
This document compares the two HR Policy RAG implementations — `02a-hr-policy-rag-chroma/` (local) and `02b-hr-policy-rag-pinecone/` (cloud) — built on identical ingestion, chunking, chain, and eval code, differing only in vector store. The goal is to document where each choice actually matters in practice, rather than treat "vector database" as an interchangeable implementation detail.

## Eval Results — Side by Side
| Metric | Chroma | Pinecone |
|---|---|---|
| Recall@4 | 100% | 100% |
| Groundedness | 4.82 / 5 | 4.82 / 5 |
| Citation Accuracy | 95% | 95% |
| Answer Relevance | 4.73 / 5 | 4.73 / 5 |
| Negative Control Refusal Accuracy | 100% | 100% |

**Key finding:** Scores are identical. At this corpus size (5 documents, ~dozens of chunks), retrieval quality is governed by embedding model and chunking strategy — not by which vector store holds the vectors. This is expected: both use the same OpenAI embeddings and the same approximate nearest-neighbor search semantics at small scale. The vector store choice does not move accuracy here.

## Where the Real Differences Are

| Dimension | Chroma (local) | Pinecone (cloud) |
|---|---|---|
| **Setup friction** | Zero — no account, no API key, runs on first `pip install` | Requires account creation, API key, index provisioning before first run |
| **Cost** | Free, unlimited (local disk/compute) | Free tier covers this project's scale; paid beyond free tier limits |
| **Reproducibility for reviewers** | Clone-and-run, no external dependency | Requires reviewer to have their own Pinecone account/index to fully reproduce |
| **Persistence** | Local `db/` folder, portable but not shared across machines | Centrally hosted, accessible from any environment with the API key |
| **Latency** | Sub-millisecond (in-process, local disk) | Network round-trip per query (~tens of ms, provider/region dependent) |
| **Scalability** | Practical ceiling in the low millions of vectors on a single machine; no built-in sharding/replication | Built for horizontal scale — sharding, replication, and multi-region managed by the provider |
| **Operational overhead** | None — no monitoring, no scaling decisions | Index management, capacity/tier planning, provider dependency for uptime |
| **Best fit** | Prototypes, small/static corpora, local-first tools, portfolio demos reviewers can run instantly | Production systems with growing/changing corpora, multi-instance deployments, or a need for managed infra |

## Takeaway
For this project's actual scale (5 HR policy documents), Chroma is the objectively better choice — it's free, has zero setup friction, and delivers identical accuracy. Pinecone's advantages (managed scaling, multi-region availability, operational hand-off) don't materialize at this size; they matter once a corpus grows into the thousands of documents, is updated frequently, or needs to serve multiple production instances concurrently.

Building both versions was intentional: it demonstrates the engineering judgment to choose infrastructure based on actual requirements rather than defaulting to "cloud is always better" — and shows hands-on familiarity with both a local-first and a managed vector database, using the same eval harness to make the comparison honest rather than anecdotal.

## Related Docs
- `02a-hr-policy-rag-chroma/docs/eval_harness.md`
- `02b-hr-policy-rag-pinecone/docs/eval_harness.md`
