"""
faiss_store.py
─────────────────────────────────────────────────────────────────────────────
Demonstrates the full lifecycle of a FAISS vector store:
    1. Build an index from documents
    2. Query with similarity search (with scores)
    3. Save the index to disk
    4. Load the index from disk
    5. Verify results are identical before and after persistence

Why persistence matters in production:
    Building a FAISS index requires embedding every document — an operation
    that costs time, money (API calls), and latency. In production, you build
    the index once, save it to disk, and load it on every subsequent startup.
    Without persistence, every application restart means re-embedding thousands
    or millions of documents: a process that might take minutes and cost dollars.
    With persistence, startup takes milliseconds and costs nothing.

Author : Pablo Suarez | github.com/SuarezPM
"""

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Directory where the FAISS index files will be persisted.
# Two files are written: index.faiss (the ANN index) and index.pkl (docstore).
INDEX_DIR = Path(__file__).resolve().parent / "faiss_index"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_and_chunk(doc_path: Path) -> list:
    """Load a text file and split it into overlapping chunks.

    Args:
        doc_path: Path to the .txt knowledge base file.

    Returns:
        List of LangChain Document chunks.
    """
    print(f"📚  Loading: {doc_path.name}")
    loader = TextLoader(str(doc_path), encoding="utf-8")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"✅  {len(chunks)} chunks ready.")
    return chunks


def _build_embeddings() -> OpenAIEmbeddings:
    """Return a configured OpenAI embeddings instance.

    We use text-embedding-3-small for cost-efficiency. The embedding model
    must be the same at index time and query time — mismatching models
    produces nonsensical similarity scores.

    Returns:
        OpenAIEmbeddings instance.
    """
    return OpenAIEmbeddings(model="text-embedding-3-small")


def _print_results(results: list, label: str) -> None:
    """Pretty-print similarity search results with scores.

    Args:
        results: List of (Document, score) tuples from similarity_search_with_score.
        label:   Label to identify the retrieval stage (before/after save).
    """
    print(f"\n  📎  Results [{label}]:")
    for i, (doc, score) in enumerate(results, 1):
        preview = doc.page_content[:100].replace("\n", " ")
        print(f"    [{i}]  score={score:.4f}  '{preview}…'")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Build
# ─────────────────────────────────────────────────────────────────────────────

def build_vectorstore(chunks: list, embeddings: OpenAIEmbeddings) -> FAISS:
    """Embed all chunks and create a FAISS vector store.

    This is the expensive step — one OpenAI API call per batch of chunks.
    In production, you run this once and then save to disk (see Stage 3).

    Args:
        chunks:     List of Document objects to embed.
        embeddings: Configured embeddings model.

    Returns:
        Populated FAISS vector store.
    """
    print("\n🔍  Building FAISS index (embedding all chunks) …")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print(f"✅  Index built — {vectorstore.index.ntotal} vectors stored.")
    return vectorstore


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Query
# ─────────────────────────────────────────────────────────────────────────────

def query_vectorstore(
    vectorstore: FAISS,
    query: str,
    k: int = 4,
) -> list:
    """Search for the K most similar chunks to the query.

    similarity_search_with_score returns (Document, score) pairs where the
    score is L2 distance by default in FAISS (lower = more similar).
    Some LangChain configurations return cosine similarity (higher = better).
    Always check the sign convention in your specific setup.

    Args:
        vectorstore: Populated FAISS index.
        query:       Natural-language query string.
        k:           Number of results to return.

    Returns:
        List of (Document, float) tuples sorted by similarity.
    """
    return vectorstore.similarity_search_with_score(query, k=k)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Save to disk
# ─────────────────────────────────────────────────────────────────────────────

def save_vectorstore(vectorstore: FAISS, index_dir: Path) -> None:
    """Persist the FAISS index to disk.

    FAISS writes two files:
        index.faiss — the ANN index (binary, fast to load)
        index.pkl   — the docstore mapping IDs → Document objects

    Why this matters:
        Rebuilding the index from scratch on every restart means re-embedding
        every document. For 10,000 chunks at $0.02/1M tokens that is about
        $0.02 — cheap — but the wall-clock time (API latency × batches) can
        take minutes. Save once, load instantly forever.

    Args:
        vectorstore: Populated FAISS index to persist.
        index_dir:   Directory path where index files will be written.

    Returns:
        None
    """
    index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))
    files = list(index_dir.iterdir())
    print(f"\n💾  Index saved to: {index_dir}")
    for f in sorted(files):
        print(f"    {f.name}  ({f.stat().st_size / 1024:.1f} KB)")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — Load from disk
# ─────────────────────────────────────────────────────────────────────────────

def load_vectorstore(index_dir: Path, embeddings: OpenAIEmbeddings) -> FAISS:
    """Load a previously saved FAISS index from disk.

    allow_dangerous_deserialization=True is required because the docstore is
    loaded via pickle. Only load indices you created yourself — never load
    an index from an untrusted source.

    Args:
        index_dir:  Directory containing index.faiss and index.pkl.
        embeddings: Must be the SAME model used when the index was built.

    Returns:
        Loaded FAISS vector store, ready for queries.
    """
    print(f"\n📂  Loading index from disk: {index_dir}")
    vectorstore = FAISS.load_local(
        str(index_dir),
        embeddings,
        allow_dangerous_deserialization=True,  # safe because WE wrote this file
    )
    print(f"✅  Loaded — {vectorstore.index.ntotal} vectors in index.")
    return vectorstore


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5 — Verify results are identical
# ─────────────────────────────────────────────────────────────────────────────

def verify_identical(results_before: list, results_after: list) -> bool:
    """Assert that two sets of retrieval results return the same documents.

    This confirms that saving and loading the index is a lossless operation —
    no chunks are lost and ranking is preserved.

    Args:
        results_before: Results from the in-memory index.
        results_after:  Results from the loaded index.

    Returns:
        True if all document contents match, False otherwise.
    """
    print("\n🔬  Verifying save/load round-trip …")
    if len(results_before) != len(results_after):
        print("❌  Different number of results!")
        return False

    for i, ((doc_b, score_b), (doc_a, score_a)) in enumerate(
        zip(results_before, results_after), 1
    ):
        if doc_b.page_content != doc_a.page_content:
            print(f"❌  Result {i} content mismatch!")
            return False

    print("✅  All results are identical — save/load round-trip verified.")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    doc_path = base_dir / "data" / "sample_docs" / "sample.txt"

    if not doc_path.exists():
        print(f"❌  Sample document not found: {doc_path}")
        sys.exit(1)

    embeddings = _build_embeddings()

    # ── Build ────────────────────────────────────────────────────────────────
    chunks = _load_and_chunk(doc_path)
    vectorstore = build_vectorstore(chunks, embeddings)

    # ── Query (before save) ───────────────────────────────────────────────────
    query = "What are the limitations of large language models?"
    print(f"\n💬  Query: '{query}'")
    results_before = query_vectorstore(vectorstore, query)
    _print_results(results_before, "in-memory index")

    # ── Save ─────────────────────────────────────────────────────────────────
    save_vectorstore(vectorstore, INDEX_DIR)

    # ── Load ─────────────────────────────────────────────────────────────────
    loaded_store = load_vectorstore(INDEX_DIR, embeddings)

    # ── Query (after load) ────────────────────────────────────────────────────
    results_after = query_vectorstore(loaded_store, query)
    _print_results(results_after, "loaded-from-disk index")

    # ── Verify ────────────────────────────────────────────────────────────────
    verify_identical(results_before, results_after)

    print("\n" + "=" * 60)
    print("  ✅  FAISS demo complete.")
    print(f"  📁  Persistent index at: {INDEX_DIR}")
    print("=" * 60)
