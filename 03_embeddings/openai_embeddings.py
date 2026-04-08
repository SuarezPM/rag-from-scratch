"""
openai_embeddings.py
─────────────────────────────────────────────────────────────────────────────
Demonstrates OpenAI text embeddings with hands-on examples:
    1. What a raw embedding vector looks like
    2. Cosine similarity between semantically similar vs dissimilar sentences
    3. Semantic arithmetic: king - man + woman ≈ queen

Purpose:
    Build intuition for what embeddings are and why they are the core primitive
    that makes semantic retrieval (and therefore RAG) possible.

Author : Pablo Suarez | github.com/SuarezPM
"""

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
import os

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# Load OPENAI_API_KEY from .env so we never hardcode secrets.
load_dotenv()

client = OpenAI()

# ─────────────────────────────────────────────────────────────────────────────
# Embedding helper
# ─────────────────────────────────────────────────────────────────────────────

def get_embedding(text: str, model: str = "text-embedding-3-small") -> np.ndarray:
    """Request an embedding vector from the OpenAI API.

    Why text-embedding-3-small:
        It costs $0.02 per million tokens — essentially free for most use cases
        — while delivering strong semantic quality. Use text-embedding-3-large
        only when you observe retrieval misses on subtle semantic distinctions.

    Args:
        text:  The text to embed. Whitespace is collapsed automatically.
        model: OpenAI embedding model name.

    Returns:
        A 1-D NumPy array of shape (1536,) for text-embedding-3-small.
    """
    # Replace newlines with spaces — the API handles them fine, but stripping
    # them avoids subtle tokenization differences between equivalent texts.
    cleaned = text.replace("\n", " ").strip()
    response = client.embeddings.create(input=[cleaned], model=model)
    return np.array(response.data[0].embedding, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Similarity computation
# ─────────────────────────────────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    Formula:
        cos(θ) = (A · B) / (||A|| × ||B||)

    Why cosine over Euclidean distance:
        Euclidean distance is sensitive to vector magnitude. Two texts that
        are semantically identical but one is twice as long might have very
        different magnitude embeddings. Cosine similarity normalises by
        magnitude so it measures direction (semantic meaning) not size.

    OpenAI embeddings are already unit-normalised, so this reduces to
    the dot product: cos(θ) = A · B. We compute the full formula anyway
    to keep the code self-explanatory.

    Args:
        a: First embedding vector.
        b: Second embedding vector.

    Returns:
        Cosine similarity in range [-1, 1]. Higher = more similar.
    """
    # Dot product
    dot = np.dot(a, b)
    # Euclidean norms
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    # Guard against division by zero for zero vectors
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


# ─────────────────────────────────────────────────────────────────────────────
# Demo 1 — Vector inspection
# ─────────────────────────────────────────────────────────────────────────────

def demo_vector_inspection() -> np.ndarray:
    """Show what a raw embedding vector looks like.

    Returns:
        The embedding vector for later use.
    """
    print("\n" + "=" * 60)
    print("  DEMO 1 — What does an embedding vector look like?")
    print("=" * 60)

    text = "Retrieval-Augmented Generation improves LLM accuracy."
    vec = get_embedding(text)

    print(f"\n  Text     : '{text}'")
    print(f"  Dimensions: {vec.shape[0]}")
    print(f"  Dtype     : {vec.dtype}")
    print(f"  Norm      : {np.linalg.norm(vec):.6f}  (≈1.0 — unit-normalised)")
    print(f"\n  First 8 values : {vec[:8].round(5).tolist()}")
    print(f"  Last  8 values : {vec[-8:].round(5).tolist()}")
    print(f"\n  💡  Each of the {vec.shape[0]} dimensions encodes some aspect of")
    print(     "     meaning — no single dimension is human-interpretable,")
    print(     "     but together they place semantically similar texts close")
    print(     "     together in this high-dimensional space.")
    return vec


# ─────────────────────────────────────────────────────────────────────────────
# Demo 2 — Semantic similarity
# ─────────────────────────────────────────────────────────────────────────────

def demo_semantic_similarity() -> None:
    """Compare cosine similarity across similar and dissimilar sentence pairs."""
    print("\n" + "=" * 60)
    print("  DEMO 2 — Semantic similarity between sentence pairs")
    print("=" * 60)

    # Three pairs ordered from most to least semantically related.
    pairs = [
        (
            "What is RAG?",
            "Can you explain Retrieval-Augmented Generation to me?",
            "Very similar — same question, different words",
        ),
        (
            "Large language models can hallucinate facts.",
            "LLMs sometimes generate incorrect information confidently.",
            "Related — same concept expressed differently",
        ),
        (
            "The capital of France is Paris.",
            "My cat enjoys sleeping in the afternoon.",
            "Unrelated — completely different topics",
        ),
    ]

    for text_a, text_b, description in pairs:
        vec_a = get_embedding(text_a)
        vec_b = get_embedding(text_b)
        score = cosine_similarity(vec_a, vec_b)

        print(f"\n  Pair    : {description}")
        print(f"  Text A  : '{text_a}'")
        print(f"  Text B  : '{text_b}'")
        print(f"  Score   : {score:.4f}  {'✅ High' if score > 0.8 else '🟡 Medium' if score > 0.5 else '❌ Low'}")

    print(f"\n  💡  Scores above ~0.85 indicate near-synonymous meaning.")
    print(     "     Scores below ~0.5 indicate unrelated content.")
    print(     "     RAG retrieval uses this score to find relevant chunks.")


# ─────────────────────────────────────────────────────────────────────────────
# Demo 3 — Semantic arithmetic
# ─────────────────────────────────────────────────────────────────────────────

def demo_semantic_arithmetic() -> None:
    """Demonstrate that semantic relationships are geometric in embedding space.

    The classic example: king - man + woman ≈ queen
    This works because the embedding model learns gender as a direction in space.
    The direction from "man" to "woman" is the same as from "king" to "queen".
    """
    print("\n" + "=" * 60)
    print("  DEMO 3 — Semantic arithmetic: king - man + woman ≈ queen?")
    print("=" * 60)

    words = ["king", "man", "woman", "queen", "prince", "princess", "doctor", "nurse"]
    print(f"\n  Fetching embeddings for: {words}")

    # Fetch all embeddings in one batch — cheaper and faster than one-by-one.
    response = client.embeddings.create(input=words, model="text-embedding-3-small")
    vecs = {
        word: np.array(response.data[i].embedding, dtype=np.float32)
        for i, word in enumerate(words)
    }

    # king - man + woman = ?
    # We expect this vector to be closest to "queen" in the vocabulary.
    result_vec = vecs["king"] - vecs["man"] + vecs["woman"]

    print("\n  Computing: king - man + woman = ?")
    print("  Ranking all words by similarity to the result vector:")

    scores = {
        word: cosine_similarity(result_vec, vec)
        for word, vec in vecs.items()
        if word not in ("king", "man", "woman")  # exclude the input words
    }
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    for rank, (word, score) in enumerate(ranked, 1):
        marker = "  👑  ← best match" if rank == 1 else ""
        print(f"    {rank}. {word:<12} {score:.4f}{marker}")

    print(f"\n  💡  If 'queen' ranks #1 or #2, the embedding space has learned")
    print(     "     that gender is a linear direction in vector space.")
    print(     "     This geometric structure is why semantic search works.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🔍  OpenAI Embeddings — Interactive Demo")
    print("   Model: text-embedding-3-small  |  Dimensions: 1,536")

    demo_vector_inspection()
    demo_semantic_similarity()
    demo_semantic_arithmetic()

    print("\n" + "=" * 60)
    print("  ✅  All demos complete.")
    print("=" * 60)
