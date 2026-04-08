# 02 — Chunking Strategies

Chunking is the step between raw documents and embeddings.  It is one of the
most underrated levers in a RAG system — getting it wrong directly degrades
retrieval quality, and no amount of prompt engineering will fix a retriever
that is returning the wrong chunks.

---

## Why chunking is critical

When a user asks a question, the retriever embeds the query and finds the
**most similar chunks** in the vector store.  The quality of the answer depends
entirely on whether the correct information landed inside a single retrievable
chunk.

- **Too small**: a chunk might contain only half of a relevant fact.
- **Too large**: a chunk contains so much unrelated text that its embedding
  is "pulled in too many directions", making it hard to match to specific queries.

---

## Strategy comparison

| Property | Small (≤200) | Medium (500) | Large (≥1000) |
|----------|-------------|--------------|---------------|
| Precision | ✅ High | ✅ Good | ⚠️ Lower |
| Context per chunk | ❌ Low | ✅ Good | ✅ High |
| Token cost per query | ✅ Low | ✅ Moderate | ❌ High |
| Best for | Short facts, FAQs | Most use cases | Narrative, themes |
| Risk | Missing multi-sentence context | — | Diluted embeddings |

### CharacterTextSplitter vs RecursiveCharacterTextSplitter

| | `CharacterTextSplitter` | `RecursiveCharacterTextSplitter` |
|---|---|---|
| Split logic | Splits on a single separator | Tries `\n\n` → `\n` → ` ` → char |
| Semantic coherence | ⚠️ Can cut mid-sentence | ✅ Respects natural boundaries |
| Recommendation | Rarely preferred | **Use this by default** |

---

## Rule of thumb

> **When in doubt, start with `chunk_size=500`, `overlap=50`,
> using `RecursiveCharacterTextSplitter`.  Tune from there.**

If retrieval misses are happening:
1. First check if the relevant information **exists** in the corpus.
2. Then try reducing chunk size (more precise retrieval).
3. Then try increasing overlap (better boundary handling).
4. Only then consider more advanced strategies (semantic chunking, etc.).

---

## When to use each strategy

```
Short, factual docs (FAQs, legal)     →  chunk_size=200, overlap=20
Technical docs, articles, reports     →  chunk_size=500, overlap=50  ← default
Long-form narrative (books, calls)    →  chunk_size=1000, overlap=100
Mixed content / unknown              →  start at 500, evaluate, adjust
```

---

## How to run

```bash
cd 02_chunking_strategies
python fixed_chunking.py
```

No API key needed — this script only loads and splits text locally.
