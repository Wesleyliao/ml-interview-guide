"""Warm-up · overlapping token chunks

### Goal & real-interview context
Split an already-tokenized document (e.g., an earnings call transcript) into fixed-size overlapping windows while recording each chunk's starting token offset for downstream citation mapping.
In a live HackerRank screen, an engineer usually states the core goal in two sentences and expects you to confirm boundary cases (`overlap >= size`, empty input, avoiding redundant trailing chunks). Here is the complete agreed spec in checklist form:

### Function signature & data shapes
```python
chunk_tokens(tokens: list[str], size: int, overlap: int) -> list[tuple[int, list[str]]]
# Example output item: (start_index, ["tok_a", "tok_b", ...])
```

### Step-by-step rules
1. **Validate config first (even if `tokens` is empty):** Require `size > 0` and `0 <= overlap < size`. Otherwise raise `ValueError`.
2. **Empty input:** If `tokens == []` (and config is valid), return `[]`.
3. **Sliding window:** Start at index `0` and advance by `stride = size - overlap` on each step.
4. **Stop condition (common trap):** Stop **as soon as an emitted chunk reaches or passes `len(tokens)`** (`start + size >= len(tokens)`). Do **not** emit a redundant trailing suffix that was already covered by the previous chunk.
5. **No mutation / no aliasing:** Each returned chunk must be a fresh list copy (`tokens[start:start + size]`), so mutating the output never mutates `tokens`.

### Worked example
```python
chunk_tokens(["a", "b", "c", "d", "e", "f", "g"], size=4, overlap=2)
# -> [(0, ["a", "b", "c", "d"]),
#     (2, ["c", "d", "e", "f"]),
#     (4, ["e", "f", "g"])]        # stops here because 4 + 4 >= 7!

chunk_tokens(["a", "b", "c", "d"], size=4, overlap=2)
# -> [(0, ["a", "b", "c", "d"])]   # 1 chunk only, NOT a second (2, ["c", "d"])
```

### Self-guided workflow (20 min)
- **Min 0–3:** Trace `len=4, size=4, overlap=2` by hand to lock in the loop stop condition.
- **Min 3–13:** Implement `chunk_tokens` below.
- **Min 13–20:** Run `check_chunking()`, then add 1–2 edge cases of your own."""

import asyncio
import math
from collections import defaultdict

def chunk_tokens(tokens: list[str], size: int, overlap: int) -> list[tuple[int, list[str]]]:
    """Return (start_index, token_slice_copy) pairs with stride = size - overlap."""
    raise NotImplementedError

def check_chunking():
    # 1. Standard overlapping window stops as soon as the end is reached
    assert chunk_tokens(list("abcdefg"), 4, 2) == [
        (0, list("abcd")),
        (2, list("cdef")),
        (4, list("efg")),
    ]

    # 2. Exact-fit input emits ONE chunk (no redundant trailing overlap)
    assert chunk_tokens(list("abcd"), 4, 2) == [(0, list("abcd"))]

    # 3. Valid empty input returns []
    assert chunk_tokens([], 4, 0) == []

    # 4. Invalid config raises ValueError EVEN when tokens is empty
    try:
        chunk_tokens([], 2, 2)
        raise AssertionError("Expected ValueError when overlap >= size on empty input")
    except ValueError:
        pass

    # 5. Output slices must not alias/mutate the caller's list
    original = list("abc")
    chunks = chunk_tokens(original, 2, 1)
    chunks[0][1][0] = "changed"
    assert original == list("abc"), "Caller input list was mutated!"

    print("Chunking samples passed!")

check_chunking()

def my_extra_chunking_tests():
    # Optional practice: add 1-2 boundary tests of your own (e.g., size=1 overlap=0, or len < size)
    pass

my_extra_chunking_tests()
