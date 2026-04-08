# 01 — Basic RAG Pipeline

A minimal, fully-commented implementation of Retrieval-Augmented Generation.
Read `simple_rag.py` top-to-bottom and you will understand every stage of the
pipeline before touching any abstraction.

---

## What `simple_rag.py` does — step by step

| Step | Function | What happens |
|------|----------|--------------|
| 1 | `load_documents()` | Reads `sample.txt` into LangChain `Document` objects |
| 2 | `split_documents()` | Breaks the document into 500-char chunks with 50-char overlap |
| 3 | `create_vectorstore()` | Embeds each chunk with OpenAI and stores vectors in FAISS |
| 4 | `build_rag_chain()` | Wires the retriever and `gpt-4o-mini` with a hallucination-prevention prompt |
| 5 | `ask()` | Takes a question → retrieves top-4 chunks → generates a grounded answer |

---

## Pipeline diagram

```
  ┌─────────────────────────────────────────────────────────────┐
  │                    INDEXING  (run once)                     │
  │                                                             │
  │  sample.txt  ──►  Chunks  ──►  Embeddings  ──►  FAISS       │
  │  (raw text)       (500c)       (1536-dim)       (index)      │
  └─────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────┐
  │                   QUERYING  (per question)                  │
  │                                                             │
  │  Question  ──►  Embed query  ──►  Top-4 chunks  ──►  LLM    │
  │                                   (retrieved)       │       │
  │                                                     ▼       │
  │                                                  Answer     │
  └─────────────────────────────────────────────────────────────┘
```

---

## How to run

```bash
# From the repo root
cd 01_basic_rag
python simple_rag.py
```

Make sure `OPENAI_API_KEY` is set in your `.env` file (copy from `.env.example`).

---

## What you will see in the terminal

```
📚  Loading documents from: .../data/sample_docs/sample.txt
✅  Loaded 1 document(s).
✅  Split into 47 chunks (avg 412 chars each).
🔍  Generating embeddings and building FAISS index …
✅  Vector store created.
✅  RAG chain assembled.

════════════════════════════════════════════════════════════
  RAG pipeline ready — asking sample questions
════════════════════════════════════════════════════════════

💬  Question: What is RAG and what problem does it solve?
────────────────────────────────────────────────────────────

🤖  Answer:
RAG (Retrieval-Augmented Generation) combines a retrieval system with
a language model. It solves the knowledge cutoff and hallucination
problems by grounding the model's answer in retrieved documents …

📎  Source chunks used:
   [1] …RAG stands for Retrieval-Augmented Generation. It was introduced…
   [2] …The retrieval component searches a vector database for the most…
   …
```

---

## Why `temperature=0`?

`temperature` controls how random the model's token sampling is.

- `temperature=1.0` → creative, varied, sometimes unpredictable
- `temperature=0.0` → fully deterministic — the model always picks the
  highest-probability next token

In RAG the model's job is to **faithfully report** what is in the retrieved
chunks, not to be creative.  Setting `temperature=0` removes unnecessary
randomness and makes answers reproducible — useful both in production and
when debugging retrieval issues.
