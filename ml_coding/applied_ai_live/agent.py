"""Mock C · a bounded tool-calling agent loop

### Goal & real-interview context
Implement a synchronous, deterministic tool-calling loop (`run_agent`) around a fake LLM and an allowlisted dictionary of tools (e.g., `get_stock_price`, `fetch_10k_section`).
Applied AI interviews at OpenAI, Anthropic, and hedge funds love this pattern because it tests state-machine clarity, message-history formatting, strict output validation before executing side effects, and loop-budget enforcement—all in ~30 lines of plain Python.

### Function signature & data shapes
```python
run_agent(question: str, model, tools: dict[str, callable], max_tool_calls: int) -> dict

# `tools` maps approved tool names -> functions taking **kwargs and returning a str:
tools = {"lookup_eps": lambda ticker, year: "$4.12"}

# `model(messages)` inspects the conversation list and returns ONE of two exact dict shapes:
# Shape 1 — Final answer:
{"type": "final", "text": "FY25 EPS was $4.12."}
# Shape 2 — Tool call request:
{"type": "tool", "name": "lookup_eps", "arguments": {"ticker": "AAPL", "year": 2025}}

# Return value of `run_agent`:
{"status": "answered", "text": "FY25 EPS was $4.12."}   # when model returns a valid final answer
{"status": "limit",    "text": None}                    # when model wants a (max_tool_calls + 1)-th tool call
```

### Conversation `messages` transcript format
1. Initialize `messages` with the user prompt:
   `[{"role": "user", "content": question}]`
2. Each time the model requests a valid, within-budget tool call, append **two** items in order:
   - Assistant turn: `{"role": "assistant", "content": response}` *(where `response` is the model's tool dict)*
   - Tool result turn: `{"role": "tool", "name": name, "content": result}` *(where `result = tools[name](**arguments)`)*
3. Pass the updated `messages` list to the next `model(messages)` call.

### Validation & budget rules (order matters!)
1. **Budget parameter check:** If `max_tool_calls < 0`, raise `ValueError` immediately. (`max_tool_calls == 0` is valid: the model gets 1 call to answer directly, with 0 tool executions allowed).
2. **Validate EVERY `response` from `model(messages)` BEFORE checking the budget limit or calling a tool:**
   - `response` must be a `dict`.
   - If `response.get("type") == "final"`: keys must be **exact** `{"type", "text"}`, and `response["text"]` must be a `str` with `response["text"].strip() != ""`. If valid, return `{"status": "answered", "text": response["text"]}`. Otherwise raise `ValueError`.
   - If `response.get("type") == "tool"`: keys must be **exact** `{"type", "name", "arguments"}`, `name` must be a `str` present in `tools`, and `arguments` must be a `dict` whose keys are all `str`. Otherwise raise `ValueError`.
   - Any other `"type"` (or missing `"type"`) raises `ValueError`.
3. **Enforce `max_tool_calls` budget:**
   - You may execute at most `max_tool_calls` tools.
   - After executing the `max_tool_calls`-th tool, the model gets **one final turn** (`max_tool_calls + 1` model calls total) to synthesize its `"final"` answer from the tool results.
   - If on that final turn the model *still* requests another (valid!) `"tool"`, **do NOT execute the tool or append to `messages`**—return `{"status": "limit", "text": None}`.
4. **Exceptions:** Let any exception raised inside `model(...)` or `tools[name](...)` propagate naturally (no retries).

### Self-guided workflow (30 min)
- **Step 1 (5 min):** Trace `max_tool_calls = 1`: `for used in range(max_tool_calls + 1):` runs for `used = 0` (1st model call -> can run 1st tool) and `used = 1` (2nd model call -> must return `"final"`, else if `"tool"` hit `"limit"`!).
- **Step 2 (15 min):** Implement `run_agent` cleanly using that `for used in range(max_tool_calls + 1):` loop.
- **Step 3 (10 min):** Run `check_agent()`, then add tests for unknown tools and malformed responses."""

import asyncio
import math
from collections import defaultdict

def run_agent(question: str, model, tools: dict, max_tool_calls: int) -> dict:
    """Run a bounded synchronous tool-calling loop for up to max_tool_calls tool executions."""
    raise NotImplementedError

def check_agent():
    calls = []

    def lookup(symbol):
        calls.append(symbol)
        return "42"

    # 1. Standard 1-tool-then-final trajectory records assistant + tool messages in order
    def model_once(messages):
        if len(messages) == 1:
            assert messages == [{"role": "user", "content": "price?"}]
            return {"type": "tool", "name": "lookup", "arguments": {"symbol": "A"}}
        assert messages[-1] == {"role": "tool", "name": "lookup", "content": "42"}
        return {"type": "final", "text": "42"}

    assert run_agent("price?", model_once, {"lookup": lookup}, max_tool_calls=1) == {
        "status": "answered",
        "text": "42",
    }
    assert calls == ["A"]

    # 2. Runaway tool loop stops at budget=2 WITHOUT executing a 3rd tool call
    calls.clear()

    def looping_model(messages):
        return {"type": "tool", "name": "lookup", "arguments": {"symbol": "A"}}

    assert run_agent("q", looping_model, {"lookup": lookup}, max_tool_calls=2) == {
        "status": "limit",
        "text": None,
    }
    assert calls == ["A", "A"], f"Expected exactly 2 tool executions, got {calls}"

    # 3. Zero tool budget still allows an immediate final answer
    assert run_agent("q", lambda m: {"type": "final", "text": "ok"}, {}, max_tool_calls=0) == {
        "status": "answered",
        "text": "ok",
    }
    print("Agent sample checks passed! Add validation-before-limit and unknown-tool tests below.")

check_agent()

def my_extra_agent_tests():
    # Practice writing 2 edge-case tests:
    # TODO 1: With max_tool_calls=0, if model returns an UNKNOWN tool {"type": "tool", "name": "hack", "arguments": {}},
    #         it must raise ValueError (validation happens BEFORE returning "limit"!).
    # TODO 2: Extra unexpected keys like {"type": "final", "text": "hi", "extra": 1} raise ValueError.
    pass

my_extra_agent_tests()
