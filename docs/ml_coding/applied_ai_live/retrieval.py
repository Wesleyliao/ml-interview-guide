"""Mock B · access-aware cosine top-k retrieval

### Goal & real-interview context
Implement an in-memory vector ranker that scores embedded research chunks against a query embedding while enforcing desk-level read permissions **before** selecting top-`k`.
In a live AI Engineer interview, building a clean, dependency-free ranking function tests Python fluency, edge-case discipline (zero vectors, negative cosine similarity, deterministic tie-breaking), and permission-boundary ordering (pre-filtering vs. post-filtering).

### Function signature & data shapes
```python
rank_documents(query: list[float], documents: list[dict], user: str, k: int) -> list[str]

# Example `query` (guaranteed non-empty, finite floats, non-zero norm):
[0.6, 0.8]

# Example item in `documents` (vectors have same length as `query`, finite floats, MAY be all zeros):
{
    "id": "nvda_10q_p4",          # unique str
    "readers": {"analyst_1"},     # set[str]
    "vector": [0.0, 5.0],         # list[float] — note: unnormalized magnitudes!
}
```

### Step-by-step rules
1. **Non-positive `k`:** If `k <= 0`, return `[]`.
2. **Pre-filter permissions:** Ignore any document where `user not in doc["readers"]`. (Never take top-`k` first and filter after—that causes underfilled results!).
3. **Skip zero vectors:** Compute Euclidean norm `sqrt(sum(x * x for x in vec))`. If a document vector has norm `0.0`, skip it (avoid `ZeroDivisionError`).
4. **Cosine similarity:** Compute `dot(query, vec) / (norm(query) * norm(vec))`. Note that negative cosine scores (down to `-1.0`) are completely valid and must **not** be dropped!
5. **Deterministic sort & top-`k`:** Rank eligible docs by:
   - `cosine_score` **descending**, then
   - `doc["id"]` **alphabetically ascending** (`"a"` before `"b"`) on exact score ties.
6. Return up to `k` document IDs (`list[str]`). Do not mutate `query` or `documents`.

### Worked example
```python
query = [1.0, 0.0]
docs = [
    {"id": "a",      "readers": {"u"}, "vector": [1.0, 0.0]},   # cos = +1.0
    {"id": "b",      "readers": {"u"}, "vector": [0.0, 1.0]},   # cos =  0.0
    {"id": "c",      "readers": {"u"}, "vector": [-1.0, 0.0]},  # cos = -1.0 (kept!)
    {"id": "zero",   "readers": {"u"}, "vector": [0.0, 0.0]},   # norm 0 -> skipped
    {"id": "secret", "readers": {"x"}, "vector": [1.0, 0.0]},   # unreadable -> skipped
]
rank_documents(query, docs, user="u", k=3)  # -> ["a", "b", "c"]
```

### Self-guided workflow (25–30 min)
- **Step 1 (15 min):** Write a clean `sort`-based baseline first (`scored.sort(key=lambda item: (-item[0], item[1]))`). Don't prematurely optimize with heaps!
- **Step 2 (10 min):** Run `check_retrieval()`, then write a test where raw dot-product `[100, 100]` disagrees with cosine `[1, 0]`."""

import asyncio
import math
from collections import defaultdict

def rank_documents(query: list[float], documents: list[dict], user: str, k: int) -> list[str]:
    """Return up to k readable non-zero-norm doc IDs ranked by (-cosine_similarity, doc_id)."""
    raise NotImplementedError

def check_retrieval():
    docs = [
        {"id": "b",      "readers": {"u"}, "vector": [0, 1]},
        {"id": "a",      "readers": {"u"}, "vector": [1, 0]},
        {"id": "secret", "readers": {"x"}, "vector": [1, 0]},
        {"id": "c",      "readers": {"u"}, "vector": [-1, 0]},
        {"id": "z",      "readers": {"u"}, "vector": [0, 0]},
    ]

    # 1. Filters unreadable & zero-norm docs; keeps negative cosine scores in order
    assert rank_documents([1, 0], docs, "u", 3) == ["a", "b", "c"]

    # 2. Non-positive k and user with no access return []
    assert rank_documents([1, 0], docs, "u", 0) == []
    assert rank_documents([1, 0], docs, "nobody", 3) == []

    # 3. Exact cosine ties break alphabetically by ID ("a" before "z")
    tied = [{"id": i, "readers": {"u"}, "vector": [1, 0]} for i in ["z", "a"]]
    assert rank_documents([1, 0], tied, "u", 1) == ["a"]

    print("Retrieval sample checks passed! Add a cosine-vs-dot-product test below.")

check_retrieval()

def my_extra_retrieval_tests():
    # Practice test: prove cosine beats large-magnitude unnormalized vectors!
    # e.g., query=[1, 0], doc "huge"=[100, 100] (dot=100, cos=0.707) vs doc "unit"=[1, 0] (dot=1, cos=1.0)
    pass

my_extra_retrieval_tests()
