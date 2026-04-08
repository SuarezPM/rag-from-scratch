# 🧠 RAG from Scratch

> **Retrieval-Augmented Generation** implemented step by step in Python — no magic, just code you can understand and extend.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.1+-1C3C3C?style=flat&logo=chainlink&logoColor=white)](https://langchain.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Author](https://img.shields.io/badge/Author-Pablo%20Suarez-blue?style=flat)](https://github.com/SuarezPM)

---

## 🤔 What is RAG and why does it matter?

Large Language Models (LLMs) like GPT-4 are powerful but have a critical limitation: **they don't know what they don't know**. Their knowledge is frozen at training time, and they hallucinate when asked about private or recent data.

**RAG solves this by giving the LLM a memory it can look things up in — at inference time.**

Instead of:
```
User question → LLM → Answer (possibly hallucinated)
```

RAG does:
```
User question → Search knowledge base → Inject relevant context → LLM → Grounded answer
```

This is the foundation of every serious enterprise AI application today.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                   INDEXING PIPELINE                  │
│                                                     │
│  Documents → Chunking → Embeddings → Vector Store   │
└─────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│                  RETRIEVAL PIPELINE                  │
│                                                     │
│  Query → Embed Query → Similarity Search → Top-K    │
└─────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│                 GENERATION PIPELINE                  │
│                                                     │
│  Context + Query → Prompt Template → LLM → Answer   │
└─────────────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
rag-from-scratch/
│
├── 01_basic_rag/
│   ├── simple_rag.py          # Minimal RAG in ~50 lines
│   └── README.md              # Explanation of each step
│
├── 02_chunking_strategies/
│   ├── fixed_chunking.py      # Split by character count
│   ├── semantic_chunking.py   # Split by meaning
│   └── README.md
│
├── 03_embeddings/
│   ├── openai_embeddings.py   # Using OpenAI Ada
│   ├── local_embeddings.py    # Using HuggingFace (free)
│   └── README.md
│
├── 04_vector_stores/
│   ├── faiss_store.py         # Local vector store
│   ├── chroma_store.py        # ChromaDB integration
│   └── README.md
│
├── 05_advanced_rag/
│   ├── reranking.py           # Improve retrieval with reranking
│   ├── hyde.py                # Hypothetical Document Embeddings
│   └── README.md
│
├── notebooks/
│   └── rag_walkthrough.ipynb  # Full interactive tutorial
│
├── data/
│   └── sample_docs/           # Sample documents for testing
│
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone and install

```bash
git clone https://github.com/SuarezPM/rag-from-scratch.git
cd rag-from-scratch
pip install -r requirements.txt
```

### 2. Set up your API key

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Run the basic RAG example

```bash
python 01_basic_rag/simple_rag.py
```

---

## 🔑 Core Concepts Covered

| Concept | What you'll learn |
|---------|------------------|
| **Document Loading** | How to ingest PDFs, text files, web pages |
| **Text Chunking** | Why chunk size matters and how to choose it |
| **Embeddings** | How text becomes numbers that capture meaning |
| **Vector Similarity** | Cosine similarity, dot product — how retrieval works |
| **Prompt Engineering** | How to inject context so the LLM uses it correctly |
| **Hallucination Prevention** | Techniques to keep answers grounded in sources |

---

## 💡 The Minimal RAG — 50 lines

Here's the core idea, stripped to its essence:

```python
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# 1. LOAD your documents
loader = TextLoader("data/my_document.txt")
docs = loader.load()

# 2. CHUNK — split into digestible pieces
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 3. EMBED — convert text to vectors
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)

# 4. RETRIEVE + GENERATE
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
)

# 5. ASK
answer = qa_chain.run("What does the document say about X?")
print(answer)
```

---

## 📚 Learning Path

This repo is structured to go from zero to production-ready:

1. **Start with** `01_basic_rag/` — understand the full pipeline end to end
2. **Then** `02_chunking_strategies/` — because chunking affects quality more than people think
3. **Then** `03_embeddings/` — understand what embeddings actually are
4. **Then** `04_vector_stores/` — local vs hosted, tradeoffs
5. **Finally** `05_advanced_rag/` — techniques used in production systems

---

## 🛠️ Requirements

```
langchain>=0.1.0
langchain-openai>=0.0.5
faiss-cpu>=1.7.4
chromadb>=0.4.0
python-dotenv>=1.0.0
tiktoken>=0.5.0
```

---

## 👤 Author

**Pablo Suarez** — AI Software Engineer  
Bridging the gap between data science research and production-ready AI systems.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat&logo=linkedin)](https://www.linkedin.com/in/suarezpm)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat&logo=github)](https://github.com/SuarezPM)

---

## 📄 License

MIT — use it, learn from it, build on it.
