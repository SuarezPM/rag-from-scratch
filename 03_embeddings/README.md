# 03 — Embeddings

Embeddings are the core primitive that makes semantic retrieval possible.
Without embeddings there is no RAG — they are how we convert human language
into a form we can do mathematics on.

---

## What are embeddings? (simple analogy)

Imagine a library where every book has a GPS coordinate instead of a Dewey
Decimal number. Books on similar topics sit physically close to each other on
the shelf. To find books related to "machine learning", you walk to the
machine-learning neighborhood and pick up everything within arm's reach.

Embeddings work exactly like this, but in 1,536 dimensions instead of 3.
The embedding model assigns every piece of text a coordinate in this
high-dimensional space. Semantically similar texts end up at nearby coordinates.
Retrieval is just: "find the coordinates closest to my query."

```
  High-dimensional space (simplified to 2D for illustration):

        ┌──────────────────────────────────────┐
        │                                      │
        │   "cat"  "kitten"                    │
        │     ●───●                            │
        │                    "automobile"      │
        │                       ●              │
        │                    "car"             │
        │                       ●              │
        │                                      │
        │                  ●  "machine learning"│
        │               ●  "deep learning"     │
        │            ●  "neural networks"      │
        │                                      │
        └──────────────────────────────────────┘
  Similar concepts cluster together.
  Unrelated concepts are far apart.
```

---

## text-embedding-3-small vs text-embedding-3-large

| | `text-embedding-3-small` | `text-embedding-3-large` |
|---|---|---|
| Dimensions | 1,536 | 3,072 |
| Quality | ✅ Excellent for most RAG | ✅✅ Best for subtle distinctions |
| Cost | $0.02 / 1M tokens | $0.13 / 1M tokens |
| Speed | Faster | Slower |
| Recommendation | **Default choice** | When retrieval quality is critical |

For most RAG applications the quality difference is imperceptible.
Start with `text-embedding-3-small` and only upgrade if you observe retrieval
misses on semantically subtle queries.

---

## What is cosine similarity?

Cosine similarity measures the **angle** between two vectors, not their
Euclidean distance. This is the right metric for embeddings because it is
invariant to vector magnitude — a 10-word sentence and a 200-word passage
about the same topic should be considered equally similar even though the
longer passage likely has a larger magnitude vector.

```
Formula:

    cos(θ) = (A · B) / (||A|| × ||B||)

    Where:
        A · B   = dot product of A and B
        ||A||   = Euclidean norm (magnitude) of A
        ||B||   = Euclidean norm of B
        θ       = angle between vectors A and B

Range:
     1.0  →  Identical meaning (angle = 0°)
     0.0  →  Unrelated (angle = 90°)
    -1.0  →  Opposite meaning (angle = 180°)

Practical thresholds (text-embedding-3-small):
    > 0.85  →  Near-synonym / paraphrase
    0.7–0.85 → Closely related
    0.5–0.7  → Somewhat related
    < 0.5   →  Likely unrelated
```

---

## How to run

```bash
cd 03_embeddings
python openai_embeddings.py
```

Requires `OPENAI_API_KEY` in your `.env` file. The script makes ~15 API calls
(all small text snippets) — total cost is under $0.001.
