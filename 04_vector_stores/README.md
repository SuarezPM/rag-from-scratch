# 04 — Vector Stores (FAISS)

A vector store is the retrieval backbone of any RAG system. It stores the
embedding vectors of every chunk and answers the question: *"given this query
vector, which chunks are most similar?"* — in milliseconds, across millions
of vectors.

---

## What is FAISS?

FAISS (Facebook AI Similarity Search) is an open-source library from Meta
Research for efficient similarity search over dense vectors. It runs entirely
in memory — no server, no config, no infrastructure. You point it at your
embeddings and it builds an index you can search instantly.

```
  How FAISS works (simplified):

  Indexing:
    chunk_1 → [0.12, -0.45, 0.88, …]  ─┐
    chunk_2 → [0.33,  0.01, 0.72, …]   ├─► FAISS index (in memory)
    chunk_3 → [-0.05, 0.91, 0.14, …]  ─┘

  Querying:
    "What is RAG?" → [0.11, -0.43, 0.85, …]
                            │
                            ▼
                     FAISS similarity search
                            │
                            ▼
                   Top-K most similar chunks
```

FAISS supports two index types relevant here:

| Index | Speed | Accuracy | Use when |
|-------|-------|----------|----------|
| `IndexFlatL2` | Slower (exact) | 100% | Small corpora, need perfect recall |
| `IndexIVFFlat` | Faster (approximate) | ~99% | Large corpora (> 100k vectors) |

LangChain's `FAISS.from_documents()` uses `IndexFlatL2` by default — exact
search, which is fine for most RAG use cases up to a few hundred thousand chunks.

---

## Why persisting the index matters in production

Building a FAISS index requires embedding every document. That means one (or
several batched) API calls to OpenAI — which costs time and money.

```
  Without persistence (naive approach):

    App starts → embed 10,000 chunks → build index → ready
                      ↑
                 ~2–5 minutes + API cost, every single restart

  With persistence (production approach):

    First run  → embed chunks → build index → save to disk
    Every restart → load index from disk → ready in < 1 second, free
```

Two files are written to disk:

```
faiss_index/
├── index.faiss   ← the ANN index (binary, fast to deserialise)
└── index.pkl     ← docstore mapping vector IDs → Document objects
```

**Rule:** always persist your FAISS index after building it. Re-embed only
when your document corpus actually changes.

---

## Comparison with other vector stores

| | FAISS | ChromaDB | Pinecone | pgvector |
|---|---|---|---|---|
| Setup | Zero | Zero | Managed SaaS | PostgreSQL ext. |
| Runs locally | ✅ | ✅ | ❌ | ✅ |
| Metadata filtering | ❌ | ✅ | ✅ | ✅ |
| Scales beyond one machine | ❌ | ❌ | ✅ | ✅ (with PG) |
| Best for | Prototyping, small prod | Dev + metadata-rich | Large prod | Already using PG |

FAISS is the right default for learning and for small-to-medium production
deployments (up to ~10M vectors on a single machine).

---

## How to run

```bash
cd 04_vector_stores
python faiss_store.py
```

Requires `OPENAI_API_KEY` in your `.env`. The script will:

1. Load and chunk `data/sample_docs/sample.txt`
2. Embed all chunks and build a FAISS index
3. Run a similarity search and print the top-4 results with scores
4. Save the index to `04_vector_stores/faiss_index/`
5. Reload the index from disk
6. Re-run the same query and verify the results are identical

Expected output:

```
📚  Loading: sample.txt
✅  47 chunks ready.

🔍  Building FAISS index (embedding all chunks) …
✅  Index built — 47 vectors stored.

💬  Query: 'What are the limitations of large language models?'

  📎  Results [in-memory index]:
    [1]  score=0.2841  'LIMITATION 1: KNOWLEDGE CUTOFF  LLMs are trained on …'
    [2]  score=0.3102  'LIMITATION 2: HALLUCINATIONS  Hallucination is the …'
    …

💾  Index saved to: .../04_vector_stores/faiss_index
    index.faiss  (47.2 KB)
    index.pkl    (18.6 KB)

📂  Loading index from disk …
✅  Loaded — 47 vectors in index.

✅  All results are identical — save/load round-trip verified.
```
