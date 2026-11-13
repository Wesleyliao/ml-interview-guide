"""Mock A · review and repair an async RAG answer path

### Goal & real-interview context
In Fundamental Equities, an AI research tool might answer an analyst's question using both public SEC filings and internal desk notes, falling back from an internal self-hosted LLM to an external commercial API if the internal cluster times out.
This exercise mirrors the **Oct 2026 Point72 GenAI first-round report (`1Point3Acres 1192502`)**: you are given a **short, broken async function** and asked to **spot the bugs, fix them, and write unit tests** using fake async dependencies.

### Function signature & data shapes
```python
await answer_query(question: str, user: str, fetch, primary, fallback) -> dict

# Document dict returned in a list by `await fetch(question)`:
{
    "id": "doc_1",               # unique str
    "text": "Q3 revenue...",     # str
    "readers": {"alice", "bob"}, # set[str] of authorized user IDs
    "external_ok": True,         # bool: True ONLY if safe to send to external fallback API
}

# Model dict returned by `await primary(question, docs)` or `await fallback(question, docs)`:
{"text": "Revenue grew 12%.", "citations": ["doc_1"]}

# Required return value of `answer_query`:
{"status": "answered", "text": str, "citations": list[str]}
# OR (when abstaining for any reason below):
{"status": "abstained"}
```

### Step-by-step rules (check the starter code against each!)
1. **Retrieve & filter access:** Call `await fetch(question)` once. Keep only docs where `user in doc["readers"]`. If **no** readable docs remain, return `{"status": "abstained"}` **without calling either model**.
2. **Call primary model:** Call `await primary(question, readable_docs)`.
3. **Fallback ONLY on `TimeoutError`:**
   - If `primary` raises `TimeoutError`, filter `readable_docs` further to keep **only** docs with `doc["external_ok"] == True`.
   - If that external-safe list is empty, return `{"status": "abstained"}` **without calling `fallback`**.
   - Otherwise call `await fallback(question, external_safe_docs)`.
4. **Do NOT swallow other errors:** Any error from `fetch`, any **non-`TimeoutError`** from `primary` (e.g., `RuntimeError`, `ValueError`, `asyncio.CancelledError`), and **any** error from `fallback` must propagate untouched.
5. **Validate model output & citations against the ACTUAL sent context:**
   - `result["text"].strip()` must be non-empty.
   - `result["citations"]` must be non-empty.
   - **Every** cited ID must belong to the exact document list passed to whichever model (`primary` or `fallback`) produced `result`. (If `fallback` answered, citing an internal-only doc that `fallback` never saw—or an unreadable doc—must abstain!).
   - If any check fails, return `{"status": "abstained"}`.

### Self-guided workflow (30–40 min)
- **Step 1 (5–8 min — Code Review):** Read the 11-line starter below *before* editing. List all 5 bugs in your head or notes (what happens on empty docs? on `ValueError`? what docs go to `fallback`? what if `text` is `"   "` or `citations` has `"secret"`?).
- **Step 2 (15 min — Patch):** Fix `answer_query` cleanly (it only takes ~18 lines of Python!).
- **Step 3 (10–15 min — Write Tests):** Run `check_rag()`, then fill in `my_extra_rag_tests()` to test unexpected errors, empty external context, and invalid citations."""

import asyncio
import math
from collections import defaultdict

async def answer_query(question: str, user: str, fetch, primary, fallback) -> dict:
    """BUGGY STARTER: Review against the 5 rules above, then repair."""
    documents = await fetch(question)
    readable = [d for d in documents if user in d["readers"]]
    try:
        result = await primary(question, readable)
    except Exception:
        result = await fallback(question, documents)
    if not result["citations"]:
        return {"status": "abstained"}
    return {"status": "answered", "text": result["text"],
            "citations": result["citations"]}

async def check_rag():
    docs = [
        {"id": "public", "text": "10-K fact", "readers": {"u"}, "external_ok": True},
        {"id": "internal", "text": "desk note", "readers": {"u"}, "external_ok": False},
        {"id": "secret", "text": "other pod", "readers": {"other"}, "external_ok": True},
    ]
    calls = []

    async def fetch(q):
        calls.append("fetch")
        return docs

    async def primary_timeout(q, context):
        calls.append(("primary", [d["id"] for d in context]))
        raise TimeoutError("internal cluster slow")

    async def fallback_ok(q, context):
        calls.append(("fallback", [d["id"] for d in context]))
        return {"text": "Grounded answer", "citations": ["public"]}

    # 1. Primary times out -> fallback receives ONLY readable + external_ok docs
    res = await answer_query("q", "u", fetch, primary_timeout, fallback_ok)
    assert res == {"status": "answered", "text": "Grounded answer", "citations": ["public"]}
    assert calls == [
        "fetch",
        ("primary", ["public", "internal"]),
        ("fallback", ["public"]),
    ], f"Unexpected call trace: {calls}"

    # 2. User with zero readable docs -> abstains immediately WITHOUT calling any model
    calls.clear()
    assert await answer_query("q", "nobody", fetch, primary_timeout, fallback_ok) == {"status": "abstained"}
    assert calls == ["fetch"], f"Models should not be called when no docs are readable: {calls}"

    # 3. Model cites an unauthorized doc ID -> abstains
    async def hallucinate_citation(q, context):
        return {"text": "Answer", "citations": ["public", "secret"]}

    assert await answer_query("q", "u", fetch, hallucinate_citation, fallback_ok) == {"status": "abstained"}
    print("RAG sample checks passed! Now implement your own regression tests below.")

asyncio.run(check_rag())

async def my_extra_rag_tests():
    # Practice writing 3 high-signal unit tests (interviewers care heavily about this!):
    # TODO 1: Primary raises RuntimeError("bug") -> error propagates, fallback is NOT called.
    # TODO 2: User only has internal docs (external_ok=False) & primary raises TimeoutError -> abstains without calling fallback.
    # TODO 3: Primary returns whitespace-only text {"text": "   ", "citations": ["public"]} -> abstains.
    pass

asyncio.run(my_extra_rag_tests())
