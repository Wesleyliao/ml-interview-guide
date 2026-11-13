"""Answer keys for authored coding mocks. Reveal after an attempt."""
import asyncio
import math
from collections import defaultdict

def chunk_tokens(tokens: list[str], size: int, overlap: int) -> list[tuple[int, list[str]]]:
    """Return (start_index, token_slice_copy) pairs with stride = size - overlap."""
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Require size > 0 and 0 <= overlap < size")
    chunks = []
    start = 0
    while start < len(tokens):
        chunks.append((start, tokens[start:start + size]))
        if start + size >= len(tokens):
            break
        start += size - overlap
    return chunks

async def answer_query(question: str, user: str, fetch, primary, fallback) -> dict:
    """Repaired async RAG path enforcing read ACLs, external egress, and citation grounding."""
    documents = await fetch(question)
    context = [d for d in documents if user in d["readers"]]
    if not context:
        return {"status": "abstained"}
    try:
        result = await primary(question, context)
    except TimeoutError:
        context = [d for d in context if d["external_ok"]]
        if not context:
            return {"status": "abstained"}
        result = await fallback(question, context)
    ids = {d["id"] for d in context}
    if (not result["text"].strip() or not result["citations"]
            or any(c not in ids for c in result["citations"])):
        return {"status": "abstained"}
    return {"status": "answered", "text": result["text"],
            "citations": result["citations"]}

def rank_documents(query: list[float], documents: list[dict], user: str, k: int) -> list[str]:
    """Return up to k readable non-zero-norm doc IDs ranked by (-cosine_similarity, doc_id)."""
    if k <= 0:
        return []
    query_norm = math.sqrt(sum(x * x for x in query))
    scored = []
    for doc in documents:
        if user not in doc["readers"]:
            continue
        vector = doc["vector"]
        norm = math.sqrt(sum(x * x for x in vector))
        if norm == 0:
            continue
        score = sum(a * b for a, b in zip(query, vector)) / (query_norm * norm)
        scored.append((score, doc["id"]))
    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    return [doc_id for _, doc_id in scored[:k]]

def run_agent(question: str, model, tools: dict, max_tool_calls: int) -> dict:
    """Run a bounded synchronous tool-calling loop for up to max_tool_calls tool executions."""
    if max_tool_calls < 0:
        raise ValueError("Tool budget must be nonnegative")
    messages = [{"role": "user", "content": question}]
    for used in range(max_tool_calls + 1):
        response = model(messages)
        if not isinstance(response, dict):
            raise ValueError("Response must be a dictionary")
        kind = response.get("type")
        if kind == "final":
            if (set(response) != {"type", "text"}
                    or not isinstance(response["text"], str)
                    or not response["text"].strip()):
                raise ValueError("Invalid final answer")
            return {"status": "answered", "text": response["text"]}
        if (kind != "tool" or set(response) != {"type", "name", "arguments"}
                or not isinstance(response["name"], str)
                or response["name"] not in tools
                or not isinstance(response["arguments"], dict)
                or not all(isinstance(k, str) for k in response["arguments"])):
            raise ValueError("Invalid tool request")
        if used == max_tool_calls:
            return {"status": "limit", "text": None}
        name = response["name"]
        messages.append({"role": "assistant", "content": response})
        result = tools[name](**response["arguments"])
        messages.append({"role": "tool", "name": name, "content": result})

def evaluate_runs(gold: dict[str, set[str]], predictions, k: int) -> dict:
    """Compute macro Precision@k, Recall@k, and MRR@k over non-empty gold queries."""
    if k <= 0:
        raise ValueError("k must be positive")
    count = sum(bool(relevant) for relevant in gold.values())
    precision = recall = reciprocal = 0.0
    for prediction in predictions:
        query_id = prediction["query_id"]
        if query_id not in gold:
            raise ValueError("Unknown query ID")
        relevant = gold[query_id]
        if not relevant:
            continue
        seen = set()
        top = []
        for doc_id in prediction["ranked_ids"]:
            if doc_id not in seen:
                seen.add(doc_id)
                top.append(doc_id)
                if len(top) == k:
                    break
        hits = sum(doc_id in relevant for doc_id in top)
        precision += hits / k
        recall += hits / len(relevant)
        reciprocal += next((1 / rank for rank, doc_id in enumerate(top, 1)
                            if doc_id in relevant), 0.0)
    if count:
        precision /= count
        recall /= count
        reciprocal /= count
    return {"queries": count, "precision": precision,
            "recall": recall, "mrr": reciprocal}
