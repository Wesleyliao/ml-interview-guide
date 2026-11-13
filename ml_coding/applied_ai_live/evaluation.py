"""Fresh mock · retrieval evaluation harness (Precision@k, Recall@k, MRR@k)

### Goal & real-interview context
Keep this exercise **unseen** until your final 60-minute timed mock!
Every Fundamental Equities AI system (search over transcripts, SEC filings, or sell-side notes) lives or dies by offline retrieval evaluation. Subtle bugs in **denominator choice, missing predictions, or duplicate ranked IDs** silently inflate metrics and mislead model decisions.

### Function signature & data shapes
```python
evaluate_runs(
    gold: dict[str, set[str]],   # query_id -> set of relevant doc_ids (may be empty set()!)
    predictions: Iterable[dict], # one-pass iterable of {"query_id": str, "ranked_ids": list[str]}
    k: int,                      # cutoff > 0
) -> dict[str, int | float]

# Return shape:
{
    "queries": int,       # count of scored queries (gold queries with NON-EMPTY relevant sets)
    "precision": float,   # macro-average Precision@k across scored queries
    "recall": float,      # macro-average Recall@k across scored queries
    "mrr": float,         # macro-average Reciprocal Rank@k across scored queries
}
```

### Step-by-step metric & edge-case rules
1. **Validate `k`:** Require `k > 0`; otherwise raise `ValueError`.
2. **Scored query denominator (`queries`):**
   - Only `query_id`s in `gold` whose relevant set is **non-empty** (`len(gold[qid]) > 0`) are scored.
   - Queries in `gold` with `set()` (empty relevance set) are **excluded** from `queries` and from all metric denominators.
   - If `queries == 0`, still consume `predictions` to validate `query_id`s (Rule 3!), then return `{"queries": 0, "precision": 0.0, "recall": 0.0, "mrr": 0.0}`.
3. **One-pass `predictions` & validation:**
   - `predictions` may be a generator/iterator (iterate over it **once**; at most one record per `query_id` is guaranteed).
   - If any prediction has a `query_id not in gold` (even when `gold` has only empty sets!), raise `ValueError`.
   - Any scored query in `gold` that is **absent** from `predictions` counts as having `ranked_ids = []` (contributing `0.0` to `precision`, `recall`, and `mrr`—do **not** drop missing predictions from the denominator!).
4. **Deduplicate BEFORE truncating to `k`:**
   - Remove duplicate `doc_id`s in `ranked_ids` keeping **first occurrence** order, *then* take the first `k` unique IDs (`top_k`). (For example, `["x", "a", "a", "b"]` with `k=3` becomes `["x", "a", "b"]`).
5. **Per-query formulas (on deduplicated `top_k`):**
   - Let `hits` = number of docs in `top_k` that appear in `gold[qid]`.
   - **`precision@k`** = `hits / k` *(always divide by `k`, even if `len(top_k) < k`!)*.
   - **`recall@k`** = `hits / len(gold[qid])`.
   - **`rr@k`** = `1 / rank` of the **first** relevant doc in `top_k` (1-indexed: rank 1 -> `1.0`, rank 2 -> `0.5`, ...), or `0.0` if `hits == 0`.
6. Macro-average `precision@k`, `recall@k`, and `rr@k` by dividing each sum by `queries`.

### Worked trace
```python
gold = {
    "q1": {"a", "b"},   # scored (2 relevant docs)
    "q2": {"c"},        # scored, MISSING from predictions -> contributes 0, 0, 0!
    "skip": set(),      # empty gold set -> ignored in denominator
}
predictions = [
    {"query_id": "q1", "ranked_ids": ["x", "a", "a", "b"]},
]
# With k = 3:
# - Scored queries = 2 ("q1" and "q2")
# - For "q1": dedup(["x", "a", "a", "b"])[:3] -> ["x", "a", "b"]
#     hits = 2 ("a" at rank 2, "b" at rank 3)
#     P@3 = 2/3,  R@3 = 2/2 = 1.0,  RR@3 = 1/2 = 0.5
# - For "q2": missing -> P@3 = 0.0, R@3 = 0.0, RR@3 = 0.0
# - Macro means over 2 scored queries:
#     precision = (2/3 + 0) / 2 = 1/3
#     recall    = (1.0 + 0) / 2 = 0.5
#     mrr       = (0.5 + 0) / 2 = 0.25
```"""

import asyncio
import math
from collections import defaultdict

def evaluate_runs(gold: dict[str, set[str]], predictions, k: int) -> dict:
    """Compute macro Precision@k, Recall@k, and MRR@k over non-empty gold queries."""
    raise NotImplementedError

def check_evaluation():
    # 1. Worked example: deduplication before top-k, missing prediction ("q2"), and empty gold ("skip")
    gold = {"q1": {"a", "b"}, "q2": {"c"}, "skip": set()}
    result = evaluate_runs(gold, iter([{"query_id": "q1", "ranked_ids": ["x", "a", "a", "b"]}]), 3)
    assert result["queries"] == 2
    assert math.isclose(result["precision"], 1 / 3)
    assert math.isclose(result["recall"], 0.5)
    assert math.isclose(result["mrr"], 0.25)

    # 2. Empty gold dictionary returns 0 queries and 0.0 metrics
    assert evaluate_runs({}, [], 1) == {"queries": 0, "precision": 0.0, "recall": 0.0, "mrr": 0.0}

    # 3. Unknown query_id in predictions still raises ValueError even when no queries are scored
    try:
        evaluate_runs({}, [{"query_id": "unknown", "ranked_ids": []}], 1)
        raise AssertionError("Expected ValueError for unknown query_id")
    except ValueError:
        pass

    print("Evaluation sample checks passed! Add short-ranking (< k) and duplicate-miss tests below.")

check_evaluation()

def my_extra_evaluation_tests():
    # Practice writing 2 edge-case tests during your mock:
    # TODO 1: Short ranking len(ranked_ids) < k still divides precision@k by k (e.g., 1 hit in 1 result for k=4 -> P@4 = 0.25).
    # TODO 2: Prediction for an empty-gold query ("skip": set()) does NOT raise ValueError and does NOT increment "queries".
    pass

my_extra_evaluation_tests()
