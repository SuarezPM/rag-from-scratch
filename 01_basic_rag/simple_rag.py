"""
simple_rag.py
─────────────────────────────────────────────────────────────────────────────
A complete, minimal RAG (Retrieval-Augmented Generation) pipeline built from
scratch using LangChain and OpenAI.

Purpose:
    Demonstrates every stage of a RAG pipeline — from raw text to a grounded
    answer — so you can understand what happens under the hood before using
    higher-level abstractions.

Author : Pablo Suarez | github.com/SuarezPM
"""

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# Standard library first, then third-party, then local.
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Load environment variables from .env so we never hardcode secrets.
load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Document loading
# ─────────────────────────────────────────────────────────────────────────────

def load_documents(file_path: str) -> list:
    """Load a plain-text file into LangChain Document objects.

    We use TextLoader because our knowledge base is a .txt file.
    For PDFs you would swap in PyPDFLoader; for web pages, WebBaseLoader.
    The abstraction stays the same — the loader just changes.

    Args:
        file_path: Absolute or relative path to the .txt file.

    Returns:
        A list of LangChain Document objects (usually one per file).
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"❌  Knowledge base not found: {file_path}")

    print(f"📚  Loading documents from: {path.resolve()}")
    loader = TextLoader(str(path), encoding="utf-8")
    documents = loader.load()
    print(f"✅  Loaded {len(documents)} document(s).")
    return documents


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Chunking
# ─────────────────────────────────────────────────────────────────────────────

def split_documents(
    documents: list,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list:
    """Split documents into smaller, overlapping chunks.

    Why overlap matters:
        Imagine a key fact that sits right at a chunk boundary — e.g., the
        subject is in chunk N and the predicate is in chunk N+1.  Without
        overlap, a query about that fact would retrieve a chunk that contains
        only half the information, leading to a poor or wrong answer.
        Overlap ensures that context spanning chunk boundaries is still
        retrievable in a single hit.

    Why RecursiveCharacterTextSplitter:
        Unlike a naive character splitter, the recursive variant tries to
        preserve natural text boundaries (paragraphs → sentences → words)
        before falling back to arbitrary character splits.  This keeps chunks
        semantically coherent, which directly improves embedding quality.

    Args:
        documents:      List of LangChain Document objects.
        chunk_size:     Maximum number of characters per chunk.
        chunk_overlap:  Number of characters shared between consecutive chunks.

    Returns:
        A list of smaller Document objects (chunks).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Split at paragraph, sentence, word — in that order of preference.
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    avg_len = sum(len(c.page_content) for c in chunks) // len(chunks) if chunks else 0
    print(f"✅  Split into {len(chunks)} chunks (avg {avg_len} chars each).")
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Embeddings + Vector store
# ─────────────────────────────────────────────────────────────────────────────

def create_vectorstore(chunks: list) -> FAISS:
    """Embed every chunk and store the vectors in a FAISS index.

    Why text-embedding-3-small:
        It is OpenAI's cheapest embedding model while still delivering
        excellent semantic quality.  Use text-embedding-3-large only if you
        notice retrieval misses on subtle semantic differences.

    Why FAISS:
        FAISS (Facebook AI Similarity Search) is an in-memory, zero-config
        vector store — perfect for prototyping and small corpora.  In
        production you would swap it for Pinecone, Weaviate, or pgvector.

    Args:
        chunks: List of Document objects to embed.

    Returns:
        A FAISS vectorstore ready for similarity search.
    """
    print("🔍  Generating embeddings and building FAISS index …")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("✅  Vector store created.")
    return vectorstore


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — RAG chain
# ─────────────────────────────────────────────────────────────────────────────

# This prompt is the single most important thing in a RAG system.
# The explicit instruction to stay within the context prevents the model from
# "filling in the gaps" with its parametric knowledge (i.e., hallucinating).
_RAG_PROMPT_TEMPLATE = """You are a knowledgeable assistant. Use ONLY the
context provided below to answer the question. Do not use any prior knowledge
or make assumptions beyond what is stated in the context.

If the answer is not contained in the context, respond exactly with:
"I don't have enough information in the provided context to answer that."

Context:
{context}

Question: {question}

Answer:"""

_RAG_PROMPT = PromptTemplate(
    template=_RAG_PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)


def build_rag_chain(vectorstore: FAISS) -> RetrievalQA:
    """Assemble the retriever + LLM into a RetrievalQA chain.

    Why temperature=0:
        In RAG we want the model to be a faithful reporter of the retrieved
        context, not a creative writer.  Temperature=0 makes the output
        deterministic and grounded — higher temperatures introduce
        unnecessary randomness that can drift away from the source material.

    Why gpt-4o-mini:
        It is fast, cheap, and capable enough for most RAG use cases.  The
        quality bottleneck in RAG is usually retrieval, not generation.

    Args:
        vectorstore: Populated FAISS index to use as the retriever.

    Returns:
        A RetrievalQA chain ready to accept questions.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # k=4 means we fetch the 4 most relevant chunks per query.
    # More chunks → more context, but also more tokens (cost ↑, latency ↑).
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",          # "stuff" concatenates all chunks into one prompt
        retriever=retriever,
        return_source_documents=True,  # we want to inspect what was retrieved
        chain_type_kwargs={"prompt": _RAG_PROMPT},
    )
    print("✅  RAG chain assembled.")
    return chain


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5 — Query
# ─────────────────────────────────────────────────────────────────────────────

def ask(chain: RetrievalQA, question: str) -> str:
    """Send a question through the RAG chain and display the result.

    Printing source documents lets you verify that the answer is grounded —
    if the retrieved chunks look wrong, you know to adjust chunking or the
    retriever's k parameter, not the prompt.

    Args:
        chain:    Assembled RetrievalQA chain.
        question: Natural-language question.

    Returns:
        The model's answer as a string.
    """
    print(f"\n💬  Question: {question}")
    print("─" * 60)

    result = chain.invoke({"query": question})
    answer = result["result"]
    sources = result["source_documents"]

    print(f"\n🤖  Answer:\n{answer}")
    print("\n📎  Source chunks used:")
    for i, doc in enumerate(sources, 1):
        preview = doc.page_content[:120].replace("\n", " ")
        print(f"   [{i}] …{preview}…")

    return answer


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Resolve the knowledge base path relative to this script's location so the
    # script works correctly regardless of the working directory it is run from.
    base_dir = Path(__file__).resolve().parent.parent
    doc_path = base_dir / "data" / "sample_docs" / "sample.txt"

    # ── Build the pipeline ──────────────────────────────────────────────────
    documents = load_documents(str(doc_path))
    chunks = split_documents(documents)
    vectorstore = create_vectorstore(chunks)
    chain = build_rag_chain(vectorstore)

    print("\n" + "═" * 60)
    print("  RAG pipeline ready — asking sample questions")
    print("═" * 60)

    # ── Ask questions ────────────────────────────────────────────────────────
    ask(chain, "What is RAG and what problem does it solve?")
    ask(chain, "What are the main limitations of large language models?")
