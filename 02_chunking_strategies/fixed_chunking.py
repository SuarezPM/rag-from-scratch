"""
fixed_chunking.py
─────────────────────────────────────────────────────────────────────────────
Compares four fixed-size chunking strategies on the same document so you can
see concretely how chunk size and splitter type affect what the retriever
will actually work with.

Why this matters:
    Chunking is the hidden lever of RAG quality.  A model can only reason
    about what is inside the retrieved chunks.  Too small → insufficient
    context per chunk; too large → irrelevant noise dilutes the signal.
    There is no universal right answer — it depends on your document type
    and query style, so comparing empirically is the correct approach.

Author : Pablo Suarez | github.com/SuarezPM
"""

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
import sys
from pathlib import Path

from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# ─────────────────────────────────────────────────────────────────────────────
# Chunking strategies
# ─────────────────────────────────────────────────────────────────────────────

# Each entry is (label, splitter_instance).
# We keep them in a list so we can iterate without repeating the comparison
# logic — adding a new strategy is a one-line change here.
def _build_strategies() -> list[tuple[str, object]]:
    """Return all splitter strategies to benchmark.

    Returns:
        List of (label, splitter) tuples in display order.
    """
    return [
        (
            "CharacterTextSplitter  — small  (200)",
            # Small chunks: retrieval is precise but each chunk may lack context.
            # Good for Q&A over dense factual documents (e.g., legal clauses).
            CharacterTextSplitter(chunk_size=200, chunk_overlap=20, separator="\n"),
        ),
        (
            "CharacterTextSplitter  — medium (500)",
            # Middle ground: usually the best starting point.
            CharacterTextSplitter(chunk_size=500, chunk_overlap=50, separator="\n"),
        ),
        (
            "CharacterTextSplitter  — large  (1000)",
            # Large chunks: more context per hit, but semantic signal is diluted
            # and you spend more tokens per query.  Good for long-form summaries.
            CharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separator="\n"),
        ),
        (
            "RecursiveCharacterTextSplitter — medium (500)",
            # The recursive variant respects paragraph and sentence boundaries
            # before falling back to character splits, so chunks stay coherent.
            # Almost always preferred over plain CharacterTextSplitter.
            RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                separators=["\n\n", "\n", " ", ""],
            ),
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Analysis helpers
# ─────────────────────────────────────────────────────────────────────────────

def _avg(values: list[int]) -> int:
    """Return integer average of a list of ints, or 0 for empty list."""
    return sum(values) // len(values) if values else 0


def _analyse(label: str, chunks: list) -> None:
    """Print a structured report for one chunking strategy.

    Args:
        label:  Human-readable strategy name.
        chunks: List of LangChain Document objects produced by the splitter.
    """
    sizes = [len(c.page_content) for c in chunks]

    print(f"\n{'─' * 60}")
    print(f"  Strategy : {label}")
    print(f"{'─' * 60}")
    print(f"  Chunks         : {len(chunks)}")
    print(f"  Avg size (chars): {_avg(sizes)}")
    print(f"  Min / Max      : {min(sizes)} / {max(sizes)}")
    print(f"\n  Preview of chunk #1:")
    preview = chunks[0].page_content[:200].replace("\n", " ") if chunks else "(empty)"
    print(f"  '{preview}…'")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Load the sample document and run all chunking strategies against it.

    Returns:
        None
    """
    base_dir = Path(__file__).resolve().parent.parent
    doc_path = base_dir / "data" / "sample_docs" / "sample.txt"

    if not doc_path.exists():
        print(f"❌  Document not found: {doc_path}")
        sys.exit(1)

    print(f"📚  Loading: {doc_path.name}")
    loader = TextLoader(str(doc_path), encoding="utf-8")
    documents = loader.load()
    total_chars = sum(len(d.page_content) for d in documents)
    print(f"✅  Document loaded — {total_chars:,} total characters.\n")

    print("=" * 60)
    print("  CHUNKING STRATEGY COMPARISON")
    print("=" * 60)

    strategies = _build_strategies()
    for label, splitter in strategies:
        chunks = splitter.split_documents(documents)
        _analyse(label, chunks)

    # ── Printed conclusion ──────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  WHEN TO USE EACH SIZE")
    print(f"{'=' * 60}")
    print("""
  📏  Small  (≤200 chars)
      ✔ Short, factual documents (FAQs, legal definitions)
      ✔ Queries that ask for a single specific fact
      ✘ Poor if answers need multi-sentence context

  📏  Medium (≈500 chars)  ← recommended default
      ✔ Best starting point for most RAG use cases
      ✔ Balances retrieval precision with answer context
      ✔ Works well for technical docs, articles, reports

  📏  Large  (≥1000 chars)
      ✔ Narrative documents (books, transcripts)
      ✔ Queries that need broad thematic context
      ✘ More irrelevant text per chunk → lower precision
      ✘ Higher token cost per query

  🔄  RecursiveCharacterTextSplitter (any size)
      ✔ Almost always better than plain CharacterTextSplitter
      ✔ Preserves paragraph / sentence boundaries
      ✔ Chunks are semantically coherent → better embeddings

  Rule of thumb: start with chunk_size=500, overlap=50,
  using RecursiveCharacterTextSplitter.  Tune from there.
""")


if __name__ == "__main__":
    main()
